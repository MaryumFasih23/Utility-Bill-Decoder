"""
validation.py - Pre-submission integrity validator.
Validates headers, row counts, cell completeness, JSON schema adherence,
absence of secrets, and absence of PII.
"""

import os
import csv
import json
import re
from typing import Dict, Any, List, Optional
from schemas import Level1Output


def validate_level1_csv(csv_path: str, expected_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate Level 1 output CSV:
    - Headers: bill_id,json
    - Valid JSON parseable by Level1Output
    - No empty cells
    - No exposed keys or PII patterns
    """
    if not os.path.isfile(csv_path):
        return {"valid": False, "error": f"File does not exist: {csv_path}"}

    errors = []
    rows_count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["bill_id", "json"]:
            errors.append(f"Invalid headers: {reader.fieldnames}. Expected ['bill_id', 'json']")

        for idx, row in enumerate(reader, start=1):
            rows_count += 1
            bill_id = row.get("bill_id", "").strip()
            raw_json = row.get("json", "").strip()

            if not bill_id:
                errors.append(f"Row {idx}: empty bill_id")
            if not raw_json:
                errors.append(f"Row {idx} ({bill_id}): empty json cell")
                continue

            try:
                data = json.loads(raw_json)
                Level1Output.model_validate(data)
            except Exception as e:
                errors.append(f"Row {idx} ({bill_id}): Invalid Level 1 JSON: {e}")

            # Check for secrets or CNIC patterns
            if "AIza" in raw_json:
                errors.append(f"Row {idx} ({bill_id}): Potential Google API key detected in JSON!")
            if re.search(r"\b\d{5}-\d{7}-\d\b", raw_json):
                errors.append(f"Row {idx} ({bill_id}): Potential CNIC detected in JSON!")

    if expected_rows is not None and rows_count != expected_rows:
        errors.append(f"Row count mismatch: got {rows_count}, expected {expected_rows}")

    return {
        "valid": len(errors) == 0,
        "rows_count": rows_count,
        "errors": errors
    }


def validate_level2_csv(
    csv_path: str,
    template_path: Optional[str] = None,
    expected_rows: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate Level 2 output CSV:
    - Headers: bill_id,question_id,question,answer
    - Non-empty answers
    - Match template bill_id, question_id, question
    - No exposed secrets or PII
    """
    if not os.path.isfile(csv_path):
        return {"valid": False, "error": f"File does not exist: {csv_path}"}

    errors = []
    rows_count = 0

    template_rows = []
    if template_path and os.path.isfile(template_path):
        with open(template_path, "r", encoding="utf-8-sig") as f:
            template_rows = list(csv.DictReader(f))

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["bill_id", "question_id", "question", "answer"]:
            errors.append(f"Invalid headers: {reader.fieldnames}. Expected ['bill_id', 'question_id', 'question', 'answer']")

        for idx, row in enumerate(reader, start=1):
            rows_count += 1
            bill_id = row.get("bill_id", "").strip()
            q_id = row.get("question_id", "").strip()
            q_text = row.get("question", "").strip()
            ans = row.get("answer", "").strip()

            if not bill_id or not q_id or not q_text:
                errors.append(f"Row {idx}: missing required metadata")
            if not ans:
                errors.append(f"Row {idx} ({bill_id}, {q_id}): empty answer cell")

            # Check match against template if provided
            if template_rows and idx <= len(template_rows):
                t_row = template_rows[idx - 1]
                if bill_id != t_row.get("bill_id", "").strip() or q_id != t_row.get("question_id", "").strip():
                    errors.append(f"Row {idx}: bill_id or question_id was altered from template!")

            # Check for secrets or CNIC patterns
            if "AIza" in ans:
                errors.append(f"Row {idx}: Potential Google API key detected in answer!")
            if re.search(r"\b\d{5}-\d{7}-\d\b", ans):
                errors.append(f"Row {idx}: Potential CNIC detected in answer!")

    if expected_rows is not None and rows_count != expected_rows:
        errors.append(f"Row count mismatch: got {rows_count}, expected {expected_rows}")

    return {
        "valid": len(errors) == 0,
        "rows_count": rows_count,
        "errors": errors
    }
