"""
csv_pipeline.py - Submission CSV generator for Level 1 and Level 2 templates.
Preserves template order, handles Windows paths, escapes correctly with utf-8 encoding.
"""

import os
import csv
import json
from typing import Dict, Any, List, Optional, Tuple

from schemas import Level1Output, InternalEvidence
from extractor import BillExtractor
from question_engine import QuestionEngine


def find_image_for_bill_id(bills_dir: str, bill_id: str) -> Optional[str]:
    """Find matching image in bills_dir with any valid image extension."""
    valid_exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    # Check exact match
    for ext in valid_exts:
        candidate = os.path.join(bills_dir, f"{bill_id}{ext}")
        if os.path.isfile(candidate):
            return candidate

    # Search through directory case-insensitively
    if os.path.isdir(bills_dir):
        for fname in os.listdir(bills_dir):
            base, ext = os.path.splitext(fname)
            if base.lower() == bill_id.lower() and ext in valid_exts:
                return os.path.join(bills_dir, fname)

    return None


class CSVPipeline:
    """
    Reads competition template CSVs, executes extraction & Q&A, and writes submission CSVs.
    """

    def __init__(
        self,
        extractor: Optional[BillExtractor] = None,
        question_engine: Optional[QuestionEngine] = None
    ):
        self.extractor = extractor or BillExtractor()
        self.question_engine = question_engine or QuestionEngine(self.extractor.client)
        # bill_id -> (Level1Output, InternalEvidence)
        self.bill_cache: Dict[str, Tuple[Level1Output, InternalEvidence]] = {}
        self.image_paths: Dict[str, str] = {}
        self.extraction_diagnostics: Dict[str, Any] = {}

    def get_or_extract_bill(self, bills_dir: str, bill_id: str) -> Tuple[Level1Output, InternalEvidence]:
        if bill_id in self.bill_cache:
            return self.bill_cache[bill_id]

        img_path = find_image_for_bill_id(bills_dir, bill_id)
        if not img_path:
            raise FileNotFoundError(f"No image found for bill_id '{bill_id}' in directory: {bills_dir}")

        l1, evidence, diag = self.extractor.extract_bill(img_path, bill_id=bill_id)
        self.bill_cache[bill_id] = (l1, evidence)
        self.image_paths[bill_id] = img_path
        self.extraction_diagnostics[bill_id] = diag
        return l1, evidence

    def process_level1(
        self,
        template_csv_path: str,
        bills_dir: str,
        output_csv_path: str
    ) -> List[Dict[str, Any]]:
        """
        Process Level 1 CSV template and write output CSV.
        Template columns: bill_id, json
        """
        if not os.path.isfile(template_csv_path):
            raise FileNotFoundError(f"Level 1 template not found: {template_csv_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        results = []

        with open(template_csv_path, mode="r", encoding="utf-8-sig") as f_in:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames or ["bill_id", "json"]
            
            rows = list(reader)
            for row in rows:
                bill_id = row["bill_id"].strip()
                l1_obj, _ = self.get_or_extract_bill(bills_dir, bill_id)
                
                # Compact, canonical JSON string
                compact_json = json.dumps(l1_obj.to_level1_dict(), separators=(",", ":"), ensure_ascii=False)
                row["json"] = compact_json
                results.append(row)

        with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(results)

        return results

    def process_level2(
        self,
        template_csv_path: str,
        bills_dir: str,
        output_csv_path: str
    ) -> List[Dict[str, Any]]:
        """
        Process Level 2 CSV template and write output CSV.
        Template columns: bill_id, question_id, question, answer
        """
        if not os.path.isfile(template_csv_path):
            raise FileNotFoundError(f"Level 2 template not found: {template_csv_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        results = []

        with open(template_csv_path, mode="r", encoding="utf-8-sig") as f_in:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames or ["bill_id", "question_id", "question", "answer"]
            
            rows = list(reader)

        # Group questions per bill (original order is preserved in `rows`)
        grouped: Dict[str, List[Tuple[str, str]]] = {}
        for idx, row in enumerate(rows):
            bill_id = (row.get("bill_id") or "").strip()
            qid = (row.get("question_id") or "").strip() or f"row{idx}"
            grouped.setdefault(bill_id, []).append((qid, (row.get("question") or "").strip()))

        answers: Dict[Tuple[str, str], str] = {}
        for bill_id, qs in grouped.items():
            l1_obj, evidence_obj = self.get_or_extract_bill(bills_dir, bill_id)
            res = self.question_engine.answer_batch(
                qs,
                l1_obj,
                evidence_obj,
                image_path=self.image_paths.get(bill_id),
                bill_id=bill_id
            )
            for qid, _ in qs:
                answers[(bill_id, qid)] = res.get(qid, "The bill does not provide enough readable information to answer this question.")

        for row in rows:
            bill_id = (row.get("bill_id") or "").strip()
            qid = (row.get("question_id") or "").strip()
            row["answer"] = answers.get((bill_id, qid), "The bill does not provide enough readable information to answer this question.")
            results.append(row)

        if "answer" not in fieldnames:
            fieldnames = list(fieldnames) + ["answer"]

        with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(results)

        return results
