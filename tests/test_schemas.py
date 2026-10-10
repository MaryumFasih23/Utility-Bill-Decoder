"""
test_schemas.py - Unit tests for Level 1 JSON schema validation and canonical formatting.
"""

import pytest
import json
from schemas import Level1Output, ChargeItem, TaxItem, _normalize_date, _normalize_month


def test_normalize_dates():
    assert _normalize_date("2026-04-03") == "2026-04-03"
    assert _normalize_date("03-04-2026") == "2026-04-03"
    assert _normalize_date("03/04/2026") == "2026-04-03"
    assert _normalize_date("03-Apr-2026") == "2026-04-03"
    assert _normalize_date("03-Apr-26") == "2026-04-03"
    assert _normalize_date("N/A") is None
    assert _normalize_date("-") is None


def test_normalize_months():
    assert _normalize_month("2026-04") == "2026-04"
    assert _normalize_month("APR-26") == "2026-04"
    assert _normalize_month("Apr-2026") == "2026-04"
    assert _normalize_month("2026-Apr") == "2026-04"
    assert _normalize_month("null") is None


def test_level1_exact_schema_dump():
    data = {
        "provider": "KE",
        "tariff": "A1-R",
        "sanctioned_load_kw": 3.0,
        "bill_month": "2026-04",
        "reading_date": "2026-04-03",
        "issue_date": "2026-04-07",
        "due_date": "2026-04-21",
        "previous_reading": 8819.0,
        "current_reading": 8970.0,
        "units_consumed": 151.0,
        "charges": [
            {"type": "fixed", "amount": 900.0},
            {"type": "energy", "amount": 1054.0},
            {"type": "energy", "amount": 663.51},
            {"type": "quarterly_adjustment", "amount": 52.91},
            {"type": "fpa", "amount": 125.27},
            {"type": "surcharge", "amount": 64.93}
        ],
        "total_charges": 2860.62,
        "taxes": [
            {"type": "electricity_duty", "amount": 29.41},
            {"type": "gst", "amount": 520.21},
            {"type": "municipal_tax", "amount": 20.0}
        ],
        "total_taxes": 569.62,
        "current_bill": 3430.24,
        "arrears": "-0.53",
        "payable_within_due_date": 3430.0,
        "payable_after_due_date": 3716.0
    }

    obj = Level1Output.model_validate(data)
    dumped = obj.to_level1_dict()

    expected_keys = [
        "provider", "tariff", "sanctioned_load_kw", "bill_month",
        "reading_date", "issue_date", "due_date",
        "previous_reading", "current_reading", "units_consumed",
        "charges", "total_charges", "taxes", "total_taxes",
        "current_bill", "arrears",
        "payable_within_due_date", "payable_after_due_date"
    ]
    assert list(dumped.keys()) == expected_keys
    assert dumped["arrears"] == -0.53
    assert dumped["sanctioned_load_kw"] == 3
    assert len(dumped["charges"]) == 6
    assert len(dumped["taxes"]) == 3
    assert dumped["charges"][1]["type"] == "energy"
    assert dumped["charges"][2]["type"] == "energy"


def test_credit_and_negative_balance_parsing():
    from schemas import parse_amount

    # CR suffix -> negative credit
    assert parse_amount("54770CR") == -54770.0
    assert parse_amount("54,770 Cr") == -54770.0
    assert parse_amount("54770 cr.") == -54770.0
    assert parse_amount("(54770)") == -54770.0
    assert parse_amount("54770-") == -54770.0
    assert parse_amount("-0.53") == -0.53

    # Non-numeric text -> None
    assert parse_amount("NOT TO BE PAID") is None
    assert parse_amount("Nil") is None
    assert parse_amount("masked") is None
    assert parse_amount("-") is None

    # LESCO/IESCO arrears age suffix: "39411 / 2" -> 39411
    assert parse_amount("39411 / 2") == 39411.0
    assert parse_amount("39411/2") == 39411.0


def test_historical_year_assignment():
    from evidence import assign_years

    # 13-month sequence from Apr 2025 to Apr 2026 (KE style)
    # Months: [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
    months = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
    res = assign_years(months, anchor=(2026, 4))
    assert len(res) == 13
    assert res[0] == "2025-04"
    assert res[8] == "2025-12"
    assert res[9] == "2026-01"
    assert res[12] == "2026-04"

