"""
test_calculations.py - Unit tests for deterministic decimal calculation routines.
"""

from decimal import Decimal
import calculations as calc


def test_tax_percentage():
    pct = calc.calculate_tax_percentage(569.62, 3430.24)
    # 569.62 / 3430.24 * 100 = 16.6058... -> 16.61%
    assert pct == Decimal("16.61")


def test_effective_rate():
    rate = calc.calculate_effective_rate_per_unit(3430.24, 151)
    # 3430.24 / 151 = 22.7168... -> 22.72
    assert rate == Decimal("22.72")


def test_consumption_history_analysis():
    history = [
        {"month": "2025-05", "units": 100},
        {"month": "2025-06", "units": 200},
        {"month": "2025-07", "units": 300},
        {"month": "2025-08", "units": 150}
    ]
    res = calc.analyze_consumption_history(history)
    assert res["count"] == 4
    assert res["total"] == 750.0
    assert res["average"] == 187.5
    assert res["min_month"] == "2025-05"
    assert res["min_units"] == 100.0
    assert res["max_month"] == "2025-07"
    assert res["max_units"] == 300.0


def test_threshold_filter():
    history = [
        {"month": "2025-05", "units": 100},
        {"month": "2025-06", "units": 250},
        {"month": "2025-07", "units": 350}
    ]
    exceeded = calc.find_months_exceeding_threshold(history, 200)
    assert len(exceeded) == 2
    assert exceeded[0] == ("2025-06", 250.0)
    assert exceeded[1] == ("2025-07", 350.0)
