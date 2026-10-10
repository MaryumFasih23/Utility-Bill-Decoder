"""
calculations.py - Deterministic mathematical routines for Level 2 Q&A using Decimal.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict, Any, Tuple


def to_decimal(val: Any) -> Optional[Decimal]:
    if val is None:
        return None
    try:
        return Decimal(str(val))
    except Exception:
        return None


def calculate_tax_percentage(total_taxes: Any, current_bill: Any) -> Optional[Decimal]:
    """Calculate taxes as a percentage of the current bill."""
    t = to_decimal(total_taxes)
    b = to_decimal(current_bill)
    if t is None or b is None or b <= Decimal("0"):
        return None
    pct = (t / b) * Decimal("100")
    return pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_charge_percentage(charge_amount: Any, total_charges: Any) -> Optional[Decimal]:
    """Calculate contribution of a charge line item to total charges."""
    c = to_decimal(charge_amount)
    tc = to_decimal(total_charges)
    if c is None or tc is None or tc <= Decimal("0"):
        return None
    pct = (c / tc) * Decimal("100")
    return pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_effective_rate_per_unit(total_amount: Any, units: Any) -> Optional[Decimal]:
    """Calculate effective cost per unit (Rs/kWh)."""
    amt = to_decimal(total_amount)
    u = to_decimal(units)
    if amt is None or u is None or u <= Decimal("0"):
        return None
    rate = amt / u
    return rate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def analyze_consumption_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze historical monthly records:
    - total units
    - average units
    - min month & units
    - max month & units
    - count of months
    """
    valid_entries = []
    for entry in history:
        m = entry.get("month")
        u = to_decimal(entry.get("units"))
        if m and u is not None:
            valid_entries.append((m, u))

    if not valid_entries:
        return {"count": 0, "total": None, "average": None, "min": None, "max": None}

    total_units = sum(u for _, u in valid_entries)
    avg_units = (total_units / Decimal(str(len(valid_entries)))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    sorted_by_units = sorted(valid_entries, key=lambda x: x[1])
    min_entry = sorted_by_units[0]
    max_entry = sorted_by_units[-1]

    return {
        "count": len(valid_entries),
        "total": float(total_units),
        "average": float(avg_units),
        "min_month": min_entry[0],
        "min_units": float(min_entry[1]),
        "max_month": max_entry[0],
        "max_units": float(max_entry[1]),
        "entries": [(m, float(u)) for m, u in valid_entries]
    }


def find_months_exceeding_threshold(
    history: List[Dict[str, Any]],
    threshold: float
) -> List[Tuple[str, float]]:
    """Return list of (month, units) where units > threshold."""
    thresh_dec = to_decimal(threshold)
    results = []
    for entry in history:
        m = entry.get("month")
        u = to_decimal(entry.get("units"))
        if m and u is not None and thresh_dec is not None and u > thresh_dec:
            results.append((m, float(u)))
    return results


def calculate_month_to_month_change(
    current_units: Any,
    previous_month_units: Any
) -> Dict[str, Any]:
    """Calculate absolute and percentage change between two months."""
    curr = to_decimal(current_units)
    prev = to_decimal(previous_month_units)
    if curr is None or prev is None:
        return {"difference": None, "percentage_change": None}
    
    diff = curr - prev
    pct = None
    if prev > Decimal("0"):
        pct = ((diff / prev) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    return {
        "difference": float(diff),
        "percentage_change": float(pct) if pct is not None else None
    }


# =========================================================================
# Deterministic fact sheet for Level 2 question answering
# =========================================================================

import calendar
import re as _re

_Q2 = Decimal("0.01")


def _fmt(d: Optional[Decimal]) -> Optional[str]:
    if d is None:
        return None
    q = d.quantize(_Q2, rounding=ROUND_HALF_UP)
    return f"{q:,.2f}"


def month_name(ym: str) -> str:
    try:
        y, m = ym.split("-")
        return f"{calendar.month_name[int(m)]} {y}"
    except Exception:
        return ym


def _series_stats(rows: List[Tuple[str, Decimal]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    total = sum(v for _, v in rows)
    avg = total / Decimal(len(rows))
    lo = min(rows, key=lambda r: r[1])
    hi = max(rows, key=lambda r: r[1])
    lows = [month_name(m) for m, v in rows if v == lo[1]]
    highs = [month_name(m) for m, v in rows if v == hi[1]]
    return {
        "months_counted": len(rows),
        "period": f"{month_name(rows[0][0])} to {month_name(rows[-1][0])}",
        "total": _fmt(total),
        "average": _fmt(avg),
        "minimum": {"value": _fmt(lo[1]), "months": lows},
        "maximum": {"value": _fmt(hi[1]), "months": highs},
        "range_max_minus_min": _fmt(hi[1] - lo[1]),
    }


def _extract_thresholds(questions: List[str]) -> List[Decimal]:
    nums: List[Decimal] = []
    for q in questions:
        for tok in _re.findall(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])", q or ""):
            try:
                d = Decimal(tok.replace(",", ""))
            except Exception:
                continue
            if 1900 <= d <= 2100 and d == d.to_integral_value():  # skip years
                continue
            if d not in nums:
                nums.append(d)
    return nums[:12]


def build_fact_sheet(level1: Dict[str, Any], history: List[Dict[str, Any]],
                     bill_month: Optional[str], questions: Optional[List[str]] = None) -> Dict[str, Any]:
    """All arithmetic needed for typical customer questions, computed with Decimal."""
    facts: Dict[str, Any] = {}
    D = to_decimal

    cb, tt, tc = D(level1.get("current_bill")), D(level1.get("total_taxes")), D(level1.get("total_charges"))
    units = D(level1.get("units_consumed"))
    arrears = D(level1.get("arrears"))
    pw, pa = D(level1.get("payable_within_due_date")), D(level1.get("payable_after_due_date"))

    cur: Dict[str, Any] = {}
    if tt is not None and cb not in (None, Decimal(0)):
        cur["taxes_pct_of_current_bill"] = _fmt(tt / cb * 100) + "%"
    if tc is not None and cb not in (None, Decimal(0)):
        cur["total_charges_pct_of_current_bill"] = _fmt(tc / cb * 100) + "%"
    if tt is not None and tc not in (None, Decimal(0)):
        cur["taxes_pct_of_total_charges"] = _fmt(tt / tc * 100) + "%"
    if units not in (None, Decimal(0)):
        if cb is not None:
            cur["current_bill_per_unit_rs"] = _fmt(cb / units)
        if tc is not None:
            cur["total_charges_per_unit_rs"] = _fmt(tc / units)
    if pw is not None and pa is not None:
        cur["late_payment_extra_rs (after - within)"] = _fmt(pa - pw)
        if pw > 0:
            cur["late_payment_extra_pct_of_within"] = _fmt((pa - pw) / pw * 100) + "%"
    if cb is not None and arrears is not None:
        cur["current_bill_plus_arrears_rs"] = _fmt(cb + arrears)
    if arrears is not None:
        cur["arrears_status"] = ("credit (negative arrears = money owed to customer)" if arrears < 0
                                 else "amount owed from previous bills" if arrears > 0 else "no arrears")
    if pw is not None:
        cur["payable_within_due_date_status"] = (
            "CREDIT balance - nothing to pay" if pw < 0 else "nothing payable" if pw == 0 else "amount to pay")

    # line-item shares
    items = []
    charges = level1.get("charges") or []
    for idx, c in enumerate(charges, 1):
        a = D(c.get("amount"))
        row = {"line": idx, "type": c.get("type"), "amount_rs": _fmt(a)}
        if tc not in (None, Decimal(0)) and a is not None:
            row["pct_of_total_charges"] = _fmt(a / tc * 100) + "%"
        if cb not in (None, Decimal(0)) and a is not None:
            row["pct_of_current_bill"] = _fmt(a / cb * 100) + "%"
        items.append(row)
    if charges:
        by_type: Dict[str, Decimal] = {}
        for c in charges:
            by_type[c["type"]] = by_type.get(c["type"], Decimal(0)) + (D(c.get("amount")) or Decimal(0))
        cur["charge_totals_by_type_rs"] = {k: _fmt(v) for k, v in by_type.items()}
        if cb not in (None, Decimal(0)):
            cur["charge_type_pct_of_current_bill"] = {k: _fmt(v / cb * 100) + "%" for k, v in by_type.items()}
        cur["sum_of_charge_lines_rs"] = _fmt(sum(by_type.values()))
    for idx, t in enumerate(level1.get("taxes") or [], 1):
        a = D(t.get("amount"))
        row = {"tax_line": idx, "type": t.get("type"), "amount_rs": _fmt(a)}
        if cb not in (None, Decimal(0)) and a is not None:
            row["pct_of_current_bill"] = _fmt(a / cb * 100) + "%"
        if tt not in (None, Decimal(0)) and a is not None:
            row["pct_of_total_taxes"] = _fmt(a / tt * 100) + "%"
        if tc not in (None, Decimal(0)) and a is not None:
            row["pct_of_total_charges"] = _fmt(a / tc * 100) + "%"
        items.append(row)
    cur["line_item_shares"] = items
    facts["current_bill_calculations"] = cur

    # history
    hist = sorted([h for h in history if h.get("month")], key=lambda h: h["month"])
    hfacts: Dict[str, Any] = {"months_available": [month_name(h["month"]) for h in hist]}
    for field in ("units", "bill_amount", "payment_amount"):
        rows = [(h["month"], D(h.get(field))) for h in hist if D(h.get(field)) is not None]
        stats: Dict[str, Any] = {}
        st = _series_stats(rows)
        if st:
            stats["all_printed_months"] = st
            if bill_month and any(m == bill_month for m, _ in rows):
                st2 = _series_stats([r for r in rows if r[0] != bill_month])
                if st2:
                    stats["excluding_current_bill_month"] = st2
            for n in (3, 6, 12):
                if len(rows) > n:
                    stats[f"last_{n}_months"] = _series_stats(rows[-n:])
            stats["ranked_high_to_low"] = [f"{month_name(m)}: {_fmt(v)}"
                                           for m, v in sorted(rows, key=lambda r: r[1], reverse=True)]
        if stats:
            hfacts[field] = stats

    mom = []
    for a, b in zip(hist, hist[1:]):
        ua, ub = D(a.get("units")), D(b.get("units"))
        if ua is None or ub is None:
            continue
        row = {"from": month_name(a["month"]), "to": month_name(b["month"]), "units_change": _fmt(ub - ua)}
        if ua != 0:
            row["pct_change"] = _fmt((ub - ua) / abs(ua) * 100) + "%"
        mom.append(row)
    if mom:
        hfacts["month_over_month_units"] = mom

    per_month = []
    for h in hist:
        u, b, p = D(h.get("units")), D(h.get("bill_amount")), D(h.get("payment_amount"))
        row: Dict[str, Any] = {"month": month_name(h["month"]), "units": _fmt(u), "bill_rs": _fmt(b),
                               "payment_rs": _fmt(p)}
        if b is not None and p is not None:
            row["payment_minus_bill_rs"] = _fmt(p - b)
        if b is not None and u not in (None, Decimal(0)):
            row["bill_per_unit_rs"] = _fmt(b / u)
        per_month.append(row)
    hfacts["per_month"] = per_month

    if bill_month and units is not None:
        y, m = map(int, bill_month.split("-"))
        prev_m = f"{y - 1 if m == 1 else y:04d}-{12 if m == 1 else m - 1:02d}"
        ly = f"{y - 1:04d}-{m:02d}"
        for label, key in (("vs_previous_month", prev_m), ("vs_same_month_last_year", ly)):
            h = next((x for x in hist if x["month"] == key), None)
            if h and D(h.get("units")) is not None:
                pu = D(h.get("units"))
                row = {"compare_month": month_name(key), "that_month_units": _fmt(pu),
                       "current_units": _fmt(units), "difference": _fmt(units - pu)}
                if pu != 0:
                    row["pct_change"] = _fmt((units - pu) / abs(pu) * 100) + "%"
                hfacts[f"current_month_{label}"] = row

    unit_rows = [(h["month"], D(h.get("units"))) for h in hist if D(h.get("units")) is not None]
    thr = []
    for t in _extract_thresholds(questions or []):
        above = [month_name(m) for m, v in unit_rows if v > t]
        below = [month_name(m) for m, v in unit_rows if v < t]
        equal = [month_name(m) for m, v in unit_rows if v == t]
        thr.append({"threshold": _fmt(t), "units_above_count": len(above), "units_above_months": above,
                    "units_below_count": len(below), "units_below_months": below,
                    "units_equal_count": len(equal)})
    if thr:
        hfacts["unit_thresholds_from_questions"] = thr
    facts["history_calculations"] = hfacts
    return facts
