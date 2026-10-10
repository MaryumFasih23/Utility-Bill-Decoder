"""
test_csv_pipeline.py - Unit tests for CSV generation and validator verification.
"""

import os
import csv
import json
import pytest
from validation import validate_level1_csv, validate_level2_csv


def test_level1_csv_validation(tmp_path):
    csv_file = tmp_path / "test_level1.csv"
    valid_data = {
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
        "charges": [],
        "total_charges": 2860.62,
        "taxes": [],
        "total_taxes": 569.62,
        "current_bill": 3430.24,
        "arrears": -0.53,
        "payable_within_due_date": 3430,
        "payable_after_due_date": 3716
    }
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bill_id", "json"])
        writer.writerow(["KESC_0008", json.dumps(valid_data)])

    res = validate_level1_csv(str(csv_file), expected_rows=1)
    assert res["valid"] is True
    assert res["rows_count"] == 1


def test_level2_csv_validation(tmp_path):
    template_file = tmp_path / "template_level2.csv"
    output_file = tmp_path / "output_level2.csv"
    
    with open(template_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bill_id", "question_id", "question", "answer"])
        writer.writerow(["KESC_0008", "q1", "What is the total units consumed?", ""])

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bill_id", "question_id", "question", "answer"])
        writer.writerow(["KESC_0008", "q1", "What is the total units consumed?", "151 units were consumed."])

    res = validate_level2_csv(str(output_file), template_path=str(template_file), expected_rows=1)
    assert res["valid"] is True
    assert res["rows_count"] == 1
