"""
main.py - Official entrypoint for WTQ 2026 Utility Bill Decoder.
Supports submission CSV pipeline execution and training mode with KESC_0008 benchmarking.
"""

import os
import sys
import argparse
import json
from typing import Optional

from model_client import GeminiModelClient, ALLOWED_MODELS, DEFAULT_MODEL
from extractor import BillExtractor
from question_engine import QuestionEngine
from csv_pipeline import CSVPipeline
from benchmark import evaluate_level1, KESC_0008_GROUND_TRUTH
from validation import validate_level1_csv, validate_level2_csv


DEFAULT_TRAIN_BILLS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "train-20261010T055743Z-1-001",
    "train",
    "bills"
)


def run_training_mode(
    bills_dir: str,
    client: GeminiModelClient,
    output_dir: str = "output/train_eval"
):
    """
    Extracts all training bills, runs KESC_0008 benchmark, and reports detailed diagnostics.
    """
    print("=" * 70)
    print("WTQ 2026 UTILITY BILL DECODER - TRAINING EVALUATION MODE")
    print(f"Bills Directory: {bills_dir}")
    print(f"Active Model:    {client.model_name}")
    print(f"Cache Directory: {client.cache_dir}")
    print("=" * 70)

    if not os.path.isdir(bills_dir):
        print(f"[ERROR] Training bills directory does not exist: {bills_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    extractor = BillExtractor(client=client)

    image_files = [
        f for f in sorted(os.listdir(bills_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Found {len(image_files)} training images: {image_files}\n")

    bench_summary = {}

    for fname in image_files:
        bill_id = os.path.splitext(fname)[0]
        img_path = os.path.join(bills_dir, fname)
        print(f"--> Extracting [{bill_id}] ({fname})...")
        
        try:
            l1_obj, evidence_obj, diags = extractor.extract_bill(img_path, bill_id=bill_id)
            l1_dict = l1_obj.to_level1_dict()
            
            # Save extracted JSON
            out_file = os.path.join(output_dir, f"{bill_id}_extracted.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(l1_dict, f, indent=2)

            # Save evidence JSON
            ev_file = os.path.join(output_dir, f"{bill_id}_evidence.json")
            with open(ev_file, "w", encoding="utf-8") as f:
                json.dump(evidence_obj.model_dump(), f, indent=2)

            print(f"    Provider: {l1_obj.provider} | Month: {l1_obj.bill_month} | Units: {l1_obj.units_consumed}")
            print(f"    Current Bill: {l1_obj.current_bill} | Charges: {len(l1_obj.charges)} items | Taxes: {len(l1_obj.taxes)} items")

            # If KESC_0008, run official ground truth evaluation
            if bill_id.upper() == "KESC_0008":
                print("\n" + "#" * 60)
                print("RUNNING OFFICIAL KESC_0008 GROUND TRUTH BENCHMARK")
                print("#" * 60)
                eval_res = evaluate_level1(l1_dict, KESC_0008_GROUND_TRUTH)
                bench_summary["KESC_0008"] = eval_res
                print(f"Accuracy Score: {eval_res['accuracy_percentage']}% ({eval_res['earned_points']}/{eval_res['total_points']} points)")
                print(f"Scalar Fields Match: {eval_res['correct_scalars_count']}/{eval_res['total_scalars_count']}")
                print(f"Charges Matched: {eval_res['charges_matched']}")
                print(f"Taxes Matched:   {eval_res['taxes_matched']}")
                if eval_res['mismatches']:
                    print("Scalar Mismatches:")
                    for m in eval_res['mismatches']:
                        print(f"  - {m['field']}: expected={m['expected']} vs actual={m['actual']}")
                if eval_res['charges_diffs']:
                    print("Charge Differences:")
                    for cd in eval_res['charges_diffs']:
                        print(f"  - {cd}")
                if eval_res['taxes_diffs']:
                    print("Tax Differences:")
                    for td in eval_res['taxes_diffs']:
                        print(f"  - {td}")
                print("#" * 60 + "\n")

        except Exception as e:
            print(f"    [ERROR] Failed to extract {bill_id}: {e}")

    # Diagnostics report
    diag_file = os.path.join(output_dir, "benchmark_report.json")
    with open(diag_file, "w", encoding="utf-8") as f:
        json.dump({
            "client_diagnostics": client.diagnostics,
            "benchmark_results": bench_summary
        }, f, indent=2)

    print(f"\nTraining evaluation complete. Output saved to: {output_dir}")
    print(f"Client Stats: Requests={client.diagnostics['requests_count']}, Cache Hits={client.diagnostics['cache_hits']}, Retries={client.diagnostics['retries_count']}")


def run_submission_pipeline(
    bills_dir: str,
    level1_template: str,
    level2_template: str,
    output_dir: str,
    client: GeminiModelClient
):
    """
    Executes the full Level 1 & Level 2 pipeline on input templates.
    """
    print("=" * 70)
    print("WTQ 2026 UTILITY BILL DECODER - SUBMISSION GENERATION PIPELINE")
    print(f"Bills Directory:    {bills_dir}")
    print(f"Level 1 Template:   {level1_template}")
    print(f"Level 2 Template:   {level2_template}")
    print(f"Output Directory:   {output_dir}")
    print(f"Active Model:       {client.model_name}")
    print("=" * 70)

    extractor = BillExtractor(client=client)
    q_engine = QuestionEngine(client=client)
    pipeline = CSVPipeline(extractor=extractor, question_engine=q_engine)

    os.makedirs(output_dir, exist_ok=True)
    out_level1_csv = os.path.join(output_dir, "level1.csv")
    out_level2_csv = os.path.join(output_dir, "level2.csv")

    # Step 1: Process Level 1 CSV
    print(f"\n[1/4] Processing Level 1 CSV: {level1_template} -> {out_level1_csv}")
    pipeline.process_level1(
        template_csv_path=level1_template,
        bills_dir=bills_dir,
        output_csv_path=out_level1_csv
    )
    print(f"      Level 1 CSV generated successfully.")

    # Step 2: Validate Level 1 CSV
    print(f"\n[2/4] Validating Level 1 CSV...")
    v1 = validate_level1_csv(out_level1_csv)
    if not v1["valid"]:
        print(f"[ERROR] Level 1 validation failed: {v1['errors']}")
        sys.exit(1)
    print(f"      Level 1 validation PASSED: {v1['rows_count']} rows valid.")

    # Step 3: Process Level 2 CSV
    print(f"\n[3/4] Processing Level 2 CSV: {level2_template} -> {out_level2_csv}")
    pipeline.process_level2(
        template_csv_path=level2_template,
        bills_dir=bills_dir,
        output_csv_path=out_level2_csv
    )
    print(f"      Level 2 CSV generated successfully.")

    # Step 4: Validate Level 2 CSV
    print(f"\n[4/4] Validating Level 2 CSV...")
    v2 = validate_level2_csv(out_level2_csv, template_path=level2_template)
    if not v2["valid"]:
        print(f"[ERROR] Level 2 validation failed: {v2['errors']}")
        sys.exit(1)
    print(f"      Level 2 validation PASSED: {v2['rows_count']} rows valid.")

    print("\n" + "=" * 70)
    print("SUBMISSION GENERATION COMPLETED SUCCESSFULLY")
    print(f"Level 1 Output: {out_level1_csv}")
    print(f"Level 2 Output: {out_level2_csv}")
    print(f"API Stats: Requests={client.diagnostics['requests_count']}, Cache Hits={client.diagnostics['cache_hits']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="WTQ 2026 Utility Bill Decoder")
    parser.add_argument("--bills", type=str, help="Path to bills image directory")
    parser.add_argument("--level1-template", type=str, help="Path to level1.csv template")
    parser.add_argument("--level2-template", type=str, help="Path to level2.csv template")
    parser.add_argument("--output", type=str, default="output", help="Directory for generated CSVs")
    parser.add_argument("--train", action="store_true", help="Run training mode on sample bills")
    parser.add_argument("--model", type=str, default=None, choices=ALLOWED_MODELS, help="Approved model selection")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cached LLM responses")

    args = parser.parse_args()

    client = GeminiModelClient(
        model_name=args.model,
        use_cache=not args.no_cache
    )

    if args.train:
        bills_path = args.bills or DEFAULT_TRAIN_BILLS_DIR
        run_training_mode(bills_dir=bills_path, client=client, output_dir=args.output)
    elif args.bills and args.level1_template and args.level2_template:
        run_submission_pipeline(
            bills_dir=args.bills,
            level1_template=args.level1_template,
            level2_template=args.level2_template,
            output_dir=args.output,
            client=client
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
