"""
benchmark.py - Official benchmark comparison against KESC_0008 ground truth.
Calculates exact scalar matches, charge/tax list matching, and numeric precision.
"""

from typing import Dict, Any, List, Tuple
import json
import math

KESC_0008_GROUND_TRUTH = {
    "provider": "KE",
    "tariff": "A1-R",
    "sanctioned_load_kw": 3,
    "bill_month": "2026-04",
    "reading_date": "2026-04-03",
    "issue_date": "2026-04-07",
    "due_date": "2026-04-21",
    "previous_reading": 8819,
    "current_reading": 8970,
    "units_consumed": 151,
    "charges": [
        {"type": "fixed", "amount": 900},
        {"type": "energy", "amount": 1054},
        {"type": "energy", "amount": 663.51},
        {"type": "quarterly_adjustment", "amount": 52.91},
        {"type": "fpa", "amount": 125.27},
        {"type": "surcharge", "amount": 64.93}
    ],
    "total_charges": 2860.62,
    "taxes": [
        {"type": "electricity_duty", "amount": 29.41},
        {"type": "gst", "amount": 520.21},
        {"type": "municipal_tax", "amount": 20}
    ],
    "total_taxes": 569.62,
    "current_bill": 3430.24,
    "arrears": -0.53,
    "payable_within_due_date": 3430,
    "payable_after_due_date": 3716
}


def compare_numeric(actual: Any, expected: Any, tol: float = 0.05) -> bool:
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    try:
        a_f = float(actual)
        e_f = float(expected)
        return abs(a_f - e_f) <= tol
    except (ValueError, TypeError):
        return False


def compare_item_lists(actual_items: List[Dict], expected_items: List[Dict]) -> Tuple[int, int, List[str]]:
    """
    Compare multiset of items (type, amount) allowing repeated items.
    Returns (matched_count, total_expected, diff_notes).
    """
    remaining_expected = list(expected_items)
    matched = 0
    diffs = []
    
    for act in actual_items:
        act_type = act.get("type")
        act_amt = act.get("amount")
        found_idx = -1
        for idx, exp in enumerate(remaining_expected):
            if exp.get("type") == act_type and compare_numeric(act_amt, exp.get("amount")):
                found_idx = idx
                break
        if found_idx >= 0:
            matched += 1
            remaining_expected.pop(found_idx)
        else:
            diffs.append(f"Unexpected or mismatched item in actual: {act}")

    for rem in remaining_expected:
        diffs.append(f"Missing expected item: {rem}")

    return matched, len(expected_items), diffs


def evaluate_level1(actual: Dict[str, Any], expected: Dict[str, Any] = KESC_0008_GROUND_TRUTH) -> Dict[str, Any]:
    """
    Detailed evaluation of an extraction dictionary against expected ground truth.
    """
    scalar_fields = [
        "provider", "tariff", "sanctioned_load_kw", "bill_month",
        "reading_date", "issue_date", "due_date",
        "previous_reading", "current_reading", "units_consumed",
        "total_charges", "total_taxes", "current_bill", "arrears",
        "payable_within_due_date", "payable_after_due_date"
    ]
    
    correct_scalars = []
    mismatched_scalars = []
    
    for f in scalar_fields:
        act_val = actual.get(f)
        exp_val = expected.get(f)
        
        if isinstance(exp_val, (int, float)):
            is_match = compare_numeric(act_val, exp_val)
        else:
            is_match = (str(act_val).strip() == str(exp_val).strip()) if (act_val and exp_val) else (act_val == exp_val)
            
        if is_match:
            correct_scalars.append(f)
        else:
            mismatched_scalars.append({
                "field": f,
                "expected": exp_val,
                "actual": act_val
            })

    # Compare charges
    charges_matched, charges_total, charges_diffs = compare_item_lists(
        actual.get("charges", []), expected.get("charges", [])
    )

    # Compare taxes
    taxes_matched, taxes_total, taxes_diffs = compare_item_lists(
        actual.get("taxes", []), expected.get("taxes", [])
    )

    # Calculate score (out of 100)
    total_scalar_points = len(scalar_fields)  # 16
    total_charge_points = charges_total       # 6
    total_tax_points = taxes_total            # 3
    total_points = total_scalar_points + total_charge_points + total_tax_points  # 25
    
    earned_points = len(correct_scalars) + charges_matched + taxes_matched
    accuracy_percentage = round((earned_points / max(1, total_points)) * 100.0, 2)

    return {
        "accuracy_percentage": accuracy_percentage,
        "earned_points": earned_points,
        "total_points": total_points,
        "correct_scalars_count": len(correct_scalars),
        "total_scalars_count": len(scalar_fields),
        "correct_scalars": correct_scalars,
        "mismatches": mismatched_scalars,
        "charges_matched": f"{charges_matched}/{charges_total}",
        "charges_diffs": charges_diffs,
        "taxes_matched": f"{taxes_matched}/{taxes_total}",
        "taxes_diffs": taxes_diffs
    }
