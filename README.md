# Women Tech Quest 2026 — Utility Bill Decoder (Build Track)

A high-accuracy, reproducible, grounded utility bill extraction and customer Q&A pipeline engineered specifically for Pakistani electricity providers (**K-Electric (KE)**, **LESCO**, and **IESCO**).

---

## 1. System Overview & Architecture

The pipeline processes electricity bills through a 4-stage architecture:
1. **Stage A: Vision Preprocessing & Quality Assessment** (`vision_utils.py`)
   - Checks image dimensions, Laplacian variance blur score, and brightness.
   - Normalizes orientation using EXIF data.
   - Automatically optimizes image resolution (up to 2400px) to maximize legibility of dense billing tables while conserving vision tokens.
2. **Stage B: High-Accuracy Structured Extraction** (`extractor.py`, `model_client.py`)
   - Queries Google's approved `gemini-3.1-flash-lite` (with `gemini-3.5-flash-lite` alternative) using the official `google-genai` Python SDK.
   - Extracts Level 1 schema while separately harvesting rich internal evidence (12-month consumption history, individual meter registers, line item descriptions, and net metering metrics) for Level 2 Q&A.
3. **Stage C: Canonical Normalization & Validation** (`schemas.py`)
   - Strict Pydantic V2 schemas enforce exact key orders, date formats (`YYYY-MM-DD` and `YYYY-MM`), negative signs for credits/arrears (`CR`), non-invented values, and filtering of zero/blank lines.
   - Strict PII redaction ensures names, CNICs, consumer numbers, meter serial numbers, and addresses are never output.
4. **Stage D: Grounded Customer Q&A Engine** (`question_engine.py`, `calculations.py`)
   - Combines deterministic Python `Decimal` arithmetic (percentages, monthly differences, threshold filtering, historical averages) with fact-grounded LLM synthesis.
   - Guaranteed bill-grounded answers without hallucinations or external tariff assumptions.
5. **Stage E: Template-Preserving CSV Pipeline & Validator** (`csv_pipeline.py`, `validation.py`)
   - Preserves row order, bill IDs, question IDs, and headers from competition templates.
   - Validates CSV output completeness and JSON validity before submission.

---

## 2. Environment & Prerequisites

- **Python Version**: `3.11+` (Tested on `Python 3.13.11` Windows 64-bit).
- **Approved Runtime LLMs**:
  - `gemini-3.1-flash-lite` (Default)
  - `gemini-3.5-flash-lite` (Configurable alternative)
- **Official SDK**: `google-genai` (v2.29.0)
- **AI-Assisted Tools Used**: **Antigravity IDE** by Google DeepMind.

---

## 3. Installation & Setup

1. **Clone/Navigate to Project Root**:
   ```bash
   cd "C:\Users\Maryam Fasih\Desktop\WQT_Anti"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```
   Inside `.env`:
   ```env
   GOOGLE_API_KEY_PRIMARY=your_primary_key_here
   GOOGLE_API_KEY_BACKUP=your_backup_key_here
   GEMINI_MODEL=gemini-3.1-flash-lite
   ```
   *(Note: Real API keys must never be committed to Git or shared).*

---

## 4. Running the Application

### A. Training & Benchmark Mode
Runs extraction across the training dataset and computes exact benchmark metrics against the official ground truth for `KESC_0008`:
```bash
python main.py --train
```

### B. Official Competition Submission Pipeline (At 12:30 PM)
When the official test images and templates arrive:
```bash
python main.py --bills "test/bills" --level1-template "test/csv/level1.csv" --level2-template "test/csv/level2.csv" --output "output"
```
This produces:
- `output/level1.csv`: Completed Level 1 JSON extractions preserving all rows.
- `output/level2.csv`: Grounded answers for all 120 test questions preserving question order.

### C. Offline Dry Run (Practice Mode)
To verify end-to-end pipeline execution on practice templates:
```bash
python create_practice_data.py
python main.py --bills "practice/bills" --level1-template "practice/csv/level1.csv" --level2-template "practice/csv/level2.csv" --output "practice_output"
```

### D. Optional Streamlit Demo Dashboard
Launch the interactive web UI for live bill inspection and Q&A:
```bash
streamlit run app.py
```

---

## 5. Automated Tests

Run the full offline test suite:
```bash
python -m pytest -v tests
```
Tests cover:
- Schema field serialization, date normalization, and key ordering (`test_schemas.py`)
- Decimal arithmetic routines and historical statistics (`test_calculations.py`)
- KESC_0008 benchmark scoring (`test_benchmark.py`)
- CSV template processing and output validation (`test_csv_pipeline.py`)

---

## 6. Key Design Decisions & Rules Compliance

- **Exact Level 1 JSON Format**: Canonical keys in identical order, null for missing/unreadable fields, empty arrays `[]` for absent itemized lines, and numbers for all amounts.
- **Negative Balances**: Retains `-` signs for CR / advance balances (e.g. `-0.53` on `KESC_0008`).
- **Repeated Line Items**: Multiple slab charges (such as dual energy charges) are kept separate rather than coalesced.
- **Robust Key Failover**: Handles HTTP 429 / 5xx with jittered exponential backoff and seamless failover between primary and backup keys.
- **Content-Hash Caching**: SHA256 hashed cache prevents duplicate bill re-processing and conserves API quota. Bypassable at any time with `--no-cache`.
- **Zero Hallucination / No Outside Data**: Customer Q&A is strictly grounded in printed bill numbers and Python-calculated metrics.

---

## 7. Troubleshooting

- **429 Resource Exhausted / Rate Limits**: The client automatically retries with backoff and switches to the backup key if available.
- **Unreadable / Masked Text**: Handled gracefully by returning `null` rather than guessing or inventing numbers.
- **Missing Images**: `find_image_for_bill_id` supports case-insensitive matching across `.png`, `.jpg`, `.jpeg` extensions.
