"""
create_practice_data.py - Generates practice templates and copies sample bills for dry run testing.
Clearly labeled as PRACTICE data, not official submission data.
"""

import os
import shutil
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_BILLS_DIR = os.path.join(BASE_DIR, "train-20261010T055743Z-1-001", "train", "bills")
PRACTICE_DIR = os.path.join(BASE_DIR, "practice")
PRACTICE_BILLS_DIR = os.path.join(PRACTICE_DIR, "bills")
PRACTICE_CSV_DIR = os.path.join(PRACTICE_DIR, "csv")

os.makedirs(PRACTICE_BILLS_DIR, exist_ok=True)
os.makedirs(PRACTICE_CSV_DIR, exist_ok=True)

# Copy the 5 training bills to practice/bills
bills_copied = []
if os.path.isdir(TRAIN_BILLS_DIR):
    for fname in os.listdir(TRAIN_BILLS_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            src = os.path.join(TRAIN_BILLS_DIR, fname)
            dst = os.path.join(PRACTICE_BILLS_DIR, fname)
            shutil.copy2(src, dst)
            bills_copied.append(os.path.splitext(fname)[0])

print(f"Copied {len(bills_copied)} bills to practice/bills: {bills_copied}")

# Create practice/csv/level1.csv
level1_path = os.path.join(PRACTICE_CSV_DIR, "level1.csv")
with open(level1_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["bill_id", "json"])
    for b_id in bills_copied:
        writer.writerow([b_id, ""])

print(f"Created practice template: {level1_path} ({len(bills_copied)} rows)")

# Create practice/csv/level2.csv with 12 diverse sample questions per bill
sample_questions = [
    ("q01", "What is the average monthly electricity consumption in units (kWh) across all printed historical months?"),
    ("q02", "In which historical month was electricity consumption the highest, and how many units were consumed?"),
    ("q03", "In which historical month was electricity consumption the lowest, and how many units were consumed?"),
    ("q04", "How many printed historical months had electricity consumption exceeding 200 units?"),
    ("q05", "What was the difference and percentage change in units consumed between the current billing month and the previous month?"),
    ("q06", "What percentage of the current month bill is made up of total taxes?"),
    ("q07", "What are the arrears or previous dues recorded on this bill, and does the consumer have an outstanding balance or a credit?"),
    ("q08", "What is the total amount payable within the due date, and what explains this amount if it is zero or a credit balance?"),
    ("q09", "What is the amount payable after the due date, and what is the late payment surcharge?"),
    ("q10", "What is the effective cost per unit of electricity in Rs/kWh for the current billing period?"),
    ("q11", "In the available billing and payment history, what were the billed amounts compared to actual payment amounts?"),
    ("q12", "What is the detailed breakdown and contribution of printed electricity charges before taxes?")
]

level2_path = os.path.join(PRACTICE_CSV_DIR, "level2.csv")
total_q_rows = 0
with open(level2_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["bill_id", "question_id", "question", "answer"])
    for b_id in bills_copied:
        for q_id_suffix, q_text in sample_questions:
            q_id = f"{b_id}_{q_id_suffix}"
            writer.writerow([b_id, q_id, q_text, ""])
            total_q_rows += 1

print(f"Created practice template: {level2_path} ({total_q_rows} rows)")
