"""
evidence.py - Deterministic normalization of historical usage / billing evidence.

The vision model reports history exactly as printed (month labels as shown,
values as printed strings). This module:
  * converts month labels to canonical YYYY-MM,
  * assigns years to year-less chart labels (KE bar charts) by anchoring the
    newest label to the bill month and walking backwards chronologically,
  * merges the usage chart, history table and billing/payment table by month,
  * sorts chronologically and records validation diagnostics.
It never invents values: anything unreadable stays None.
"""

from typing import Any, Dict, List, Optional, Tuple

from schemas import (
    MonthlyHistoryEntry,
    _normalize_month,
    month_label_to_number,
    parse_amount,
)


def _shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def assign_years(month_numbers: List[int], anchor: Tuple[int, int]) -> List[Optional[str]]:
    """
    Assign years to an ordered (oldest -> newest) list of month numbers.
    The newest label is placed at or before the anchor (bill month).
    """
    if not month_numbers:
        return []
    anchor_year, anchor_month = anchor
    out: List[Optional[str]] = [None] * len(month_numbers)
    last = month_numbers[-1]
    year = anchor_year if last <= anchor_month else anchor_year - 1
    out[-1] = f"{year:04d}-{last:02d}"
    for i in range(len(month_numbers) - 2, -1, -1):
        cur, nxt = month_numbers[i], month_numbers[i + 1]
        if cur >= nxt:
            year -= 1
        out[i] = f"{year:04d}-{cur:02d}"
    return out


def _anchor_from_bill_month(bill_month: Optional[str]) -> Optional[Tuple[int, int]]:
    if not bill_month:
        return None
    try:
        y, m = bill_month.split("-")
        return int(y), int(m)
    except Exception:
        return None


def _resolve_months(labels: List[Any], anchor: Optional[Tuple[int, int]]) -> List[Optional[str]]:
    """Resolve labels to YYYY-MM; uses explicit years when printed, else anchored inference."""
    explicit = [_normalize_month(str(l)) if l is not None else None for l in labels]
    if all(e is not None for e in explicit) and explicit:
        return explicit
    nums = [month_label_to_number(l) for l in labels]
    if anchor and all(n is not None for n in nums) and nums:
        inferred = assign_years(nums, anchor)  # type: ignore[arg-type]
        # Prefer explicit where present
        return [e or i for e, i in zip(explicit, inferred)]
    return explicit


def build_history(raw: Dict[str, Any], bill_month: Optional[str]) -> Tuple[List[MonthlyHistoryEntry], Dict[str, Any]]:
    """Merge raw chart/table reads into a chronological, validated history."""
    anchor = _anchor_from_bill_month(bill_month)
    merged: Dict[str, Dict[str, Any]] = {}
    conflicts: List[str] = []
    unresolved: List[str] = []

    def put(month: Optional[str], field: str, value: Any, source: str, label: Any):
        if month is None:
            unresolved.append(f"{source}:{label}")
            return
        entry = merged.setdefault(month, {"month": month, "sources": []})
        if source not in entry["sources"]:
            entry["sources"].append(source)
        if value is None:
            return
        if entry.get(field) is None:
            entry[field] = value
        elif field in ("units", "bill_amount", "payment_amount") and abs(float(entry[field]) - float(value)) > 0.5:
            conflicts.append(f"{month}.{field}: kept {entry[field]} ({entry['sources'][0]}), other source {value} ({source})")

    # 1) Text tables first (more legible than chart labels)
    table = [r for r in (raw.get("history_table") or []) if isinstance(r, dict)]
    t_months = _resolve_months([r.get("month") for r in table], anchor)
    for r, mon in zip(table, t_months):
        put(mon, "units", parse_amount(r.get("units")), "history_table", r.get("month"))
        put(mon, "bill_amount", parse_amount(r.get("bill")), "history_table", r.get("month"))
        put(mon, "payment_amount", parse_amount(r.get("payment")), "history_table", r.get("month"))
        status = (r.get("status") or "").strip() if isinstance(r.get("status"), str) else None
        if status and mon:
            merged[mon]["status"] = status

    bp = [r for r in (raw.get("billing_payment_table") or []) if isinstance(r, dict)]
    b_months = _resolve_months([r.get("month") for r in bp], anchor)
    for r, mon in zip(bp, b_months):
        put(mon, "bill_amount", parse_amount(r.get("billed")), "billing_payment_table", r.get("month"))
        put(mon, "payment_amount", parse_amount(r.get("payment")), "billing_payment_table", r.get("month"))
        if mon and r.get("pay_date"):
            merged[mon]["pay_date"] = str(r.get("pay_date")).strip()

    # 2) Usage bar chart (labels often lack years -> anchored inference)
    chart = [r for r in (raw.get("usage_chart") or []) if isinstance(r, dict)]
    c_labels = [r.get("label") for r in chart]
    c_months = _resolve_months(c_labels, anchor)
    year_hint_mismatch = 0
    for r, mon in zip(chart, c_months):
        hint = r.get("year_hint")
        if mon and hint and str(hint).strip()[-2:].isdigit() and not mon.startswith(("20" + str(hint).strip()[-2:])):
            year_hint_mismatch += 1
        put(mon, "units", parse_amount(r.get("units")), "usage_chart", r.get("label"))

    entries = [MonthlyHistoryEntry.model_validate(merged[k]) for k in sorted(merged.keys())]

    # Validation diagnostics
    months = [e.month for e in entries]
    missing: List[str] = []
    if months:
        y, m = map(int, months[0].split("-"))
        last = months[-1]
        cur = f"{y:04d}-{m:02d}"
        guard = 0
        while cur < last and guard < 60:
            if cur not in merged:
                missing.append(cur)
            y, m = _shift_month(y, m, 1)
            cur = f"{y:04d}-{m:02d}"
            guard += 1
    future = [mm for mm in months if bill_month and mm > bill_month]
    diagnostics = {
        "record_count": len(entries),
        "records_with_units": sum(1 for e in entries if e.units is not None),
        "first_month": months[0] if months else None,
        "last_month": months[-1] if months else None,
        "chronological": months == sorted(months),
        "missing_months_in_range": missing,
        "months_after_bill_month": future,
        "chart_bars_read": len(chart),
        "history_table_rows_read": len(table),
        "billing_payment_rows_read": len(bp),
        "chart_year_hint_mismatches": year_hint_mismatch,
        "source_conflicts": conflicts,
        "unresolved_labels": unresolved,
    }
    return entries, diagnostics
