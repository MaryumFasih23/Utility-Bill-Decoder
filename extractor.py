"""
extractor.py - Staged bill extraction engine producing Level 1 JSON and Internal Evidence.

Stage A: image quality inspection + resolution normalization
Stage B: full-image structured extraction (Level 1 + evidence + section boxes)
Stage C1: targeted crop/upscale re-read of the history section (Level 2 evidence)
Stage C2: arithmetic *diagnostics* -> targeted re-read of only the flagged printed fields
Stage D: strict Pydantic normalization (dates, numbers, enums, exact keys)

Arithmetic is only used to decide whether to re-read; a value is replaced only by
another *printed* reading of the same field, never by a computed value.
"""

from typing import Tuple, Dict, Any, Optional, List
import json
import re
import os

from schemas import Level1Output, InternalEvidence, parse_amount
from model_client import GeminiModelClient
from vision_utils import (
    load_image, image_to_bytes, optimize_image_for_llm, inspect_image_quality,
    crop_box_1000, ensure_min_dimension,
)
from evidence import build_history

PROMPT_VERSION = "extract-v2.1"

EXTRACTION_PROMPT = """You are a meticulous reader of Pakistani electricity bills (KE / K-Electric, LESCO, IESCO).
Read ONLY what is printed. Never calculate, guess, or "fix" values to make totals balance.

PRIVACY: never output names, CNICs, addresses, consumer/account/contract numbers, reference numbers,
meter serial numbers, invoice numbers or phone numbers.

GENERAL RULES
- Copy every monetary amount EXACTLY as printed, as a JSON STRING, keeping minus signs, commas and any
  "CR" suffix (e.g. "54770CR", "-0.53", "-91,877.02", "39411 / 2"). The software parses them.
- "CR" after an amount means CREDIT (money owed TO the customer). Keep the "CR" text.
- If a field is blank, masked, unreadable or not printed, use null. Do not infer it from other fields.
- Text such as "NOT TO BE PAID" is NOT an amount: report it in the *_printed_text evidence fields and use
  null for the numeric field.
- Dates: output YYYY-MM-DD. bill_month: YYYY-MM.

LEVEL 1 FIELDS
- provider: "KE", "LESCO" or "IESCO".
- tariff: the tariff CODE only, exactly as printed (e.g. "A1-R", "A-1a(01)", "A-2a(04)"). Do not append the
  tariff category words such as DOMESTIC/COMMERCIAL.
- sanctioned_load_kw: printed sanctioned load ("Sanc Load"/"SAN LOAD"), number only.
- previous_reading / current_reading: only when the bill shows a single meter-reading row. If several register
  rows are printed (peak/off-peak/import/export/net-metering), use null for both (record rows in evidence).
- units_consumed: the billed units for the month as explicitly printed (e.g. KE "Current Month 151 units", the
  UNITS of a single meter row). If several register rows exist and no single billed-units figure is clearly
  labelled, use null.
- charges: one entry per individually printed charge line with a non-zero amount, in printed order.
  Types: energy (variable/unit/slab charges; keep each slab line separate), fixed (fixed charges),
  fpa (FCA / FPA / fuel adjustment, incl. "Total FPA"), quarterly_adjustment (QTA / uniform quarterly
  adjustment; keep repeated lines separate), surcharge (additional/PHL/debt surcharge), meter_rent,
  subsidy (subsidies / relief; ALWAYS negative, e.g. a printed "Subsidies 3332" -> "-3332"), other.
  Summary-style bills (typical LESCO/IESCO "BILL CHARGES BREAKDOWN"): do NOT list aggregate rows such as
  "Total Electricity Charges", "Net Electricity Charges", "Taxes", "Current Bill", "Grand Total" as charge items.
  Only list genuinely separate component lines (e.g. Subsidies, FPA). If none, charges = [].
- total_charges: KE -> the printed "Electricity Charges" subtotal. LESCO/IESCO -> the printed
  "Total Electricity Charges" value.
- taxes: individually printed tax lines only. gst (Sales Tax/GST), electricity_duty, income_tax,
  municipal_tax (MUCT/KMC/municipal/PTV fee), other_tax (extra/further/retailer tax). If only one combined
  "Taxes" figure is printed, taxes = [] and put that figure in total_taxes.
- total_taxes: printed "Taxes and Duties" / "Taxes" total.
- current_bill: printed "Electricity Charges for current month" / "Current Bill".
- arrears: printed previous dues / arrears (KE "Previous Dues"). Keep negative signs / CR. LESCO/IESCO may print
  an age suffix like "39411 / 2" - copy it as printed.
- payable_within_due_date: printed "Amount Payable within Due Date" / "PAYABLE WITHIN DUE DATE" (KE "Amount Due").
  Keep "CR" if printed.
- payable_after_due_date: the printed TOTAL payable after the due date. If several late totals are printed, take
  the highest. L.P. surcharge amounts are NOT totals. If no total is printed (blank, "NOT TO BE PAID"), null.

EVIDENCE (for later customer questions; never part of Level 1)
- line_item_details: EVERY row of the charges/bill-calculation section, including subtotals, balances,
  payments, net-metering benefit and outstanding balance: {"description", "units", "rate", "amount", "section"}.
- meter_registers: every meter-reading row: {"register_name", "previous_reading", "current_reading", "units", "mdi_kw"}.
- net_metering_present (true if bill mentions net metering / import-export / prosumer), net_metering_units_imported,
  net_metering_units_exported, net_metering_net_units, net_metering_benefit_pkr, outstanding_balance_pkr.
- subsidies_or_relief_pkr, fpa_amount_pkr, late_payment_surcharge_pkr (highest printed L.P. surcharge).
- payable_within_due_date_printed_text, payable_after_due_date_printed_text: exact printed text.
- notes: short factual notes printed on the bill relevant to billing (FPA notes, subsidy notes, net metering notes).
- history_present: true if a usage/billing history chart or table is printed.
- history_expected_months: number of months shown in the history chart/table (e.g. 12 or 13), else null.
- section_boxes: bounding boxes [ymin, xmin, ymax, xmax] normalized to 0-1000 for the sections that exist:
  "history" (usage chart AND billing/payment history tables together), "financial_summary" (amounts payable,
  arrears, current bill, due date boxes), "charges" (charge breakdown / bill calculation table), "meter".

Return ONE JSON object:
{
  "level1": {"provider":..., "tariff":..., "sanctioned_load_kw":..., "bill_month":..., "reading_date":...,
             "issue_date":..., "due_date":..., "previous_reading":..., "current_reading":..., "units_consumed":...,
             "charges":[{"type":..., "amount":"..."}], "total_charges":"...", "taxes":[{"type":..., "amount":"..."}],
             "total_taxes":"...", "current_bill":"...", "arrears":"...", "payable_within_due_date":"...",
             "payable_after_due_date":"..."},
  "internal_evidence": {"line_item_details":[...], "meter_registers":[...], "net_metering_present":false,
             "net_metering_units_imported":null, "net_metering_units_exported":null, "net_metering_net_units":null,
             "net_metering_benefit_pkr":null, "outstanding_balance_pkr":null, "subsidies_or_relief_pkr":null,
             "fpa_amount_pkr":null, "late_payment_surcharge_pkr":null,
             "payable_within_due_date_printed_text":null, "payable_after_due_date_printed_text":null,
             "notes":[], "history_present":true, "history_expected_months":null,
             "section_boxes":{"history":[0,0,0,0]}}
}
"""

HISTORY_PROMPT = """This image is (part of) a Pakistani electricity bill showing consumption / billing history.
Transcribe EVERY history value exactly as printed. Do not calculate or guess. Unreadable -> null.

1. usage_chart: if there is a bar chart of monthly units (e.g. KE "13 Month Usage History"), list EVERY bar from
   LEFT to RIGHT: {"label": month label as printed (e.g. "Apr"), "year_hint": the year printed above that part
   of the chart if any (e.g. "2025"), "units": the number printed on/in the bar as a string}.
   Count the bars carefully; a 13-month chart has 13 bars.
2. history_table: if there is a table with MONTH / UNITS / BILL / PAYMENT columns (LESCO/IESCO "BILL HISTORY"),
   list EVERY row. Two-column layouts: read the LEFT block top-to-bottom, then the RIGHT block top-to-bottom.
   {"month": as printed (e.g. "AUG 25", "Sep25"), "status": text or null, "units": string, "bill": string,
    "payment": string or null if blank}. Keep minus signs (net-metering bills can have negative units and bills).
3. billing_payment_table: if there is a "Billing & Payment History" table (MM/YY, Billed Amount, Pay-Date, Payment),
   list every row: {"month": as printed (e.g. "03/26"), "billed": string, "pay_date": string, "payment": string}.

Return JSON: {"usage_chart": [...], "history_table": [...], "billing_payment_table": [...]}
Use [] for any structure that is not present."""

VERIFY_PROMPT_TEMPLATE = """Re-read the following fields from this Pakistani electricity bill image VERY carefully,
exactly as printed. Do not calculate. Copy amounts as strings exactly as printed, keeping minus signs, commas and any
"CR" suffix. If a field is not printed / blank / text such as "NOT TO BE PAID", return null.
Field definitions:
{definitions}
Return JSON: {{"fields": {{{keys}}}, "evidence": {{"<field>": "the printed label and text you read"}}}}"""

FIELD_DEFINITIONS = {
    "current_bill": "current month bill ('Current Bill' / 'Electricity Charges for current month')",
    "arrears": "arrears / previous dues (may have CR or an age suffix like '39411 / 2')",
    "payable_within_due_date": "total amount payable within due date (keep CR if printed)",
    "payable_after_due_date": "highest printed total payable after due date (not the surcharge itself)",
    "total_charges": "KE 'Electricity Charges' subtotal / LESCO-IESCO 'Total Electricity Charges'",
    "total_taxes": "total taxes ('Taxes and Duties' / 'Taxes')",
    "previous_reading": "previous meter reading (single meter row only)",
    "current_reading": "present/current meter reading (single meter row only)",
    "units_consumed": "units billed this month",
    "issue_date": "issue date (YYYY-MM-DD)",
    "due_date": "due date (YYYY-MM-DD)",
    "reading_date": "reading date (YYYY-MM-DD)",
    "charges": "list of individually printed charge lines [{\"type\": energy|fixed|fpa|quarterly_adjustment|surcharge|meter_rent|subsidy|other, \"amount\": \"...\"}] (exclude subtotals; subsidies negative)",
    "taxes": "list of individually printed tax lines [{\"type\": gst|electricity_duty|income_tax|municipal_tax|other_tax, \"amount\": \"...\"}] (exclude totals)",
}

MONEY_SUMMARY_FIELDS = {"current_bill", "arrears", "payable_within_due_date", "payable_after_due_date"}


def _parse_json(text: str) -> Dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start >= 0 and end > start:
            return json.loads(clean[start:end + 1])
        raise


def _tolerant_evidence(fields: Dict[str, Any]) -> InternalEvidence:
    """Validate evidence; drop malformed list rows / fields instead of failing the bill."""
    from schemas import MeterRegisterEntry, DetailedChargeEntry
    clean: Dict[str, Any] = {}
    item_models = {"meter_registers": MeterRegisterEntry, "line_item_details": DetailedChargeEntry}
    for k, v in fields.items():
        if k in item_models:
            rows = []
            for row in v if isinstance(v, list) else []:
                try:
                    rows.append(item_models[k].model_validate(row))
                except Exception:
                    continue
            clean[k] = rows
        elif k == "notes":
            clean[k] = [str(n) for n in v] if isinstance(v, list) else []
        else:
            clean[k] = v
    try:
        return InternalEvidence.model_validate(clean)
    except Exception:
        ok: Dict[str, Any] = {}
        for k, v in clean.items():
            try:
                InternalEvidence.model_validate({k: v})
                ok[k] = v
            except Exception:
                continue
        return InternalEvidence.model_validate(ok)


def consistency_flags(l1: Level1Output) -> List[Dict[str, Any]]:
    """Arithmetic/format diagnostics. Returns list of {check, fields, detail}. Never modifies values."""
    flags: List[Dict[str, Any]] = []
    tol = 2.0
    charges_sum = sum(c.amount for c in l1.charges)
    subsidy_sum = sum(c.amount for c in l1.charges if c.type == "subsidy")
    fpa_sum = sum(c.amount for c in l1.charges if c.type == "fpa")

    if l1.payable_within_due_date is not None and l1.current_bill is not None and l1.arrears is not None:
        expected = [l1.current_bill + l1.arrears, l1.current_bill + l1.arrears + fpa_sum]
        p = l1.payable_within_due_date
        ok = min(abs(p - e) for e in expected) <= tol or (p == 0 and min(expected) <= 0)
        if not ok:
            flags.append({"check": "payable_vs_current_plus_arrears",
                          "fields": ["current_bill", "arrears", "payable_within_due_date"],
                          "detail": f"payable={p}, current+arrears={expected[0]:.2f}"})
    if l1.payable_within_due_date is not None and l1.payable_after_due_date is not None:
        pw, pa = l1.payable_within_due_date, l1.payable_after_due_date
        if (pw > 0 and pa < pw) or (pw < 0 and pa > 0):
            flags.append({"check": "after_due_vs_within_due",
                          "fields": ["payable_within_due_date", "payable_after_due_date"],
                          "detail": f"within={pw}, after={pa}"})
    if l1.total_charges is not None and l1.total_taxes is not None and l1.current_bill is not None:
        cands = [l1.total_charges + l1.total_taxes, l1.total_charges + subsidy_sum + l1.total_taxes]
        if min(abs(l1.current_bill - c) for c in cands) > tol:
            flags.append({"check": "charges_plus_taxes_vs_current_bill",
                          "fields": ["total_charges", "total_taxes", "current_bill"],
                          "detail": f"current_bill={l1.current_bill}, charges+taxes={cands[0]:.2f}"})
    if len(l1.charges) >= 3 and l1.total_charges is not None and abs(charges_sum - l1.total_charges) > 1.0:
        flags.append({"check": "charge_items_vs_total_charges", "fields": ["charges", "total_charges"],
                      "detail": f"sum={charges_sum:.2f}, total={l1.total_charges}"})
    if len(l1.taxes) >= 2 and l1.total_taxes is not None:
        tsum = sum(t.amount for t in l1.taxes)
        if abs(tsum - l1.total_taxes) > 1.0:
            flags.append({"check": "tax_items_vs_total_taxes", "fields": ["taxes", "total_taxes"],
                          "detail": f"sum={tsum:.2f}, total={l1.total_taxes}"})
    if None not in (l1.previous_reading, l1.current_reading, l1.units_consumed):
        if abs((l1.current_reading - l1.previous_reading) - l1.units_consumed) > 1.0:
            flags.append({"check": "readings_vs_units",
                          "fields": ["previous_reading", "current_reading", "units_consumed"],
                          "detail": f"{l1.current_reading}-{l1.previous_reading} != {l1.units_consumed}"})
    if l1.issue_date and l1.due_date and l1.due_date < l1.issue_date:
        flags.append({"check": "due_before_issue", "fields": ["issue_date", "due_date"], "detail": ""})
    if l1.reading_date and l1.issue_date and l1.reading_date > l1.issue_date:
        flags.append({"check": "reading_after_issue", "fields": ["reading_date", "issue_date"], "detail": ""})
    return flags


class BillExtractor:
    """Staged extraction pipeline (see module docstring)."""

    def __init__(self, client: Optional[GeminiModelClient] = None):
        self.client = client or GeminiModelClient()

    # ------------------------------------------------------------------ helpers
    def _generate_json(self, prompt: str, img) -> Dict[str, Any]:
        img_bytes = image_to_bytes(img, format="PNG")
        raw = self.client.generate(prompt=prompt, image_bytes=img_bytes,
                                   image_mime_type="image/png", response_mime_type="application/json")
        return _parse_json(raw)

    def _history_pass(self, img, boxes: Dict[str, Any], expected: Optional[int],
                      bill_month: Optional[str]) -> Tuple[list, Dict[str, Any]]:
        diag: Dict[str, Any] = {"history_crop_used": False}
        crop = crop_box_1000(img, boxes.get("history"), pad=0.03) if boxes else None
        target = ensure_min_dimension(crop, 1800) if crop is not None else img
        diag["history_crop_used"] = crop is not None
        raw = self._generate_json(HISTORY_PROMPT, target)
        entries, hdiag = build_history(raw, bill_month)
        diag.update(hdiag)
        # Validation: if a crop under-recovers months vs. the printed count, re-read on the full image once.
        if crop is not None and expected and hdiag["records_with_units"] < max(1, int(expected) - 1):
            raw_full = self._generate_json(HISTORY_PROMPT, img)
            entries_full, hdiag_full = build_history(raw_full, bill_month)
            diag["history_full_image_retry"] = True
            if hdiag_full["records_with_units"] > hdiag["records_with_units"]:
                entries, diag = entries_full, {**diag, **hdiag_full, "history_crop_used": False}
        return entries, diag

    def _verify_pass(self, img, l1_raw: Dict[str, Any], l1: Level1Output, boxes: Dict[str, Any],
                     flags: List[Dict[str, Any]]) -> Tuple[Level1Output, Dict[str, Any]]:
        fields = sorted({f for fl in flags for f in fl["fields"]})
        diag: Dict[str, Any] = {"flags": flags, "reread_fields": fields, "replaced": {}}
        use_crop = set(fields) <= MONEY_SUMMARY_FIELDS and boxes and boxes.get("financial_summary")
        target = crop_box_1000(img, boxes.get("financial_summary"), pad=0.03) if use_crop else None
        target = ensure_min_dimension(target, 1600) if target is not None else img
        diag["verify_crop_used"] = target is not img
        prompt = VERIFY_PROMPT_TEMPLATE.format(
            definitions="\n".join(f"- {k}: {FIELD_DEFINITIONS.get(k, k)}" for k in fields),
            keys=", ".join(f'"{k}": ...' for k in fields),
        )
        reread = self._generate_json(prompt, target).get("fields", {}) or {}
        candidate_raw = dict(l1_raw)
        for k in fields:
            if k in reread:
                candidate_raw[k] = reread[k]
        try:
            candidate = Level1Output.model_validate(candidate_raw)
        except Exception as e:
            diag["reread_error"] = str(e)
            return l1, diag
        # Accept the re-read only if it is at least as consistent; both are printed readings.
        if len(consistency_flags(candidate)) <= len(flags):
            before, after = l1.to_level1_dict(), candidate.to_level1_dict()
            diag["replaced"] = {k: {"from": before[k], "to": after[k]} for k in fields if before[k] != after[k]}
            diag["remaining_flags"] = consistency_flags(candidate)
            return candidate, diag
        diag["reread_rejected"] = True
        return l1, diag

    # --------------------------------------------------------------- main API
    def extract_bill(self, image_path: str, bill_id: Optional[str] = None
                     ) -> Tuple[Level1Output, InternalEvidence, Dict[str, Any]]:
        if not bill_id:
            bill_id = os.path.splitext(os.path.basename(image_path))[0]

        # Stage A
        quality_info = inspect_image_quality(image_path)
        img = ensure_min_dimension(optimize_image_for_llm(load_image(image_path), max_dimension=2400), 1600)

        # Stage B
        parsed = self._generate_json(EXTRACTION_PROMPT, img)
        raw_l1 = parsed.get("level1", {}) or {}
        raw_ev = parsed.get("internal_evidence", {}) or {}
        level1_obj = Level1Output.model_validate(raw_l1)
        boxes = raw_ev.get("section_boxes") or {}
        if not isinstance(boxes, dict):
            boxes = {}

        diagnostics: Dict[str, Any] = {"bill_id": bill_id, "prompt_version": PROMPT_VERSION,
                                       "quality": quality_info}

        # Stage C2: diagnostics-driven targeted re-read (only when something looks wrong)
        flags = consistency_flags(level1_obj)
        if flags:
            level1_obj, vdiag = self._verify_pass(img, raw_l1, level1_obj, boxes, flags)
            diagnostics["verification"] = vdiag
        else:
            diagnostics["verification"] = {"flags": []}

        # Evidence (Stage D validation)
        ev_fields = {k: v for k, v in raw_ev.items()
                     if k in InternalEvidence.model_fields and k not in ("section_boxes", "consumption_history")}
        evidence_obj = _tolerant_evidence(ev_fields)
        evidence_obj.section_boxes = {k: v for k, v in boxes.items() if isinstance(v, list) and len(v) == 4}
        evidence_obj.bill_id = bill_id
        evidence_obj.provider = level1_obj.provider
        evidence_obj.bill_month = level1_obj.bill_month
        evidence_obj.sanctioned_load_kw = level1_obj.sanctioned_load_kw

        # Stage C1: history re-read on a targeted crop
        if raw_ev.get("history_present", True):
            try:
                entries, hdiag = self._history_pass(img, boxes, raw_ev.get("history_expected_months"),
                                                    level1_obj.bill_month)
                evidence_obj.consumption_history = entries
                evidence_obj.history_diagnostics = {**hdiag,
                                                    "expected_months_printed": raw_ev.get("history_expected_months")}
                if level1_obj.bill_month:
                    evidence_obj.history_includes_current_month = any(
                        e.month == level1_obj.bill_month for e in entries)
            except Exception as e:  # history failure must not break Level 1
                evidence_obj.history_diagnostics = {"error": str(e)}

        diagnostics["history"] = evidence_obj.history_diagnostics
        diagnostics["remaining_flags"] = consistency_flags(level1_obj)
        return level1_obj, evidence_obj, diagnostics
