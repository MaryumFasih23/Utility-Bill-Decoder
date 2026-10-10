"""
question_engine.py - Bill-grounded Question Answering Engine for Level 2.

Design (generalizes to unseen question texts):
1. Python computes a deterministic Decimal "fact sheet" (history stats, month-over-month changes,
   thresholds mentioned in the questions, line-item shares, credit/payable status).
2. One Gemini call per bill answers all of that bill's questions together, grounded in:
   Level 1 facts + internal evidence + the fact sheet + the bill image (for re-inspection of
   anything not captured in evidence). The model is told to take numbers from the fact sheet
   rather than doing arithmetic itself.
3. Every answer is validated: non-empty, PII-scrubbed, and each number is checked against the
   evidence/fact sheet; unsupported numbers are logged in diagnostics.
4. Missing answers fall back to a per-question call, then to an explicit "not available" answer.
"""

from typing import Dict, Any, Optional, List, Tuple
import json
import re

from schemas import Level1Output, InternalEvidence
from model_client import GeminiModelClient
import calculations as calc

QA_PROMPT_VERSION = "qa-v2.0"

QA_PROMPT = """You answer customer questions about ONE Pakistani electricity bill ({provider}, bill month {bill_month_name}).
Use ONLY the bill data below and the attached bill image. No outside tariffs, regulations or assumptions.

RULES
1. Start with the direct answer, then a short justification. 1-3 sentences, plain English.
2. For ANY arithmetic (percentages, averages, totals, differences, counts, per-unit costs), use the numbers in
   CALCULATIONS. They were computed exactly in Python. Only if a needed calculation is absent, compute it yourself,
   show the formula briefly, and say it is an estimate.
3. Always include units: electricity in "units (kWh)", money as "Rs. 1,234.56", months as "April 2026".
4. Negative amounts / "CR" mean a CREDIT owed to the customer. A negative or zero payable amount means nothing
   has to be paid; explain it using the printed arrears/credit/net-metering figures.
5. History questions: say which months were used (e.g. "April 2025 to April 2026, 13 months"). If the question's
   period is ambiguous (e.g. "last 12 months" when 13 months are printed), state the assumption you used.
6. If the information is not printed or not readable, say clearly that the bill does not show it. Never invent
   numbers, months or line items.
7. Never mention internal terms like JSON, fields, evidence or "calculations section". Never include names,
   CNICs, addresses, account/reference/meter numbers.

BILL FACTS (official extraction):
{level1}

ADDITIONAL BILL EVIDENCE (history oldest->newest, line items, meter registers, notes):
{evidence}

CALCULATIONS (exact, Python Decimal):
{facts}

QUESTIONS:
{questions}

Return JSON: {{"answers": [{{"question_id": "...", "answer": "..."}}]}} with one entry per question, same ids."""

_PII_PATTERNS = [
    re.compile(r"\b\d{5}-\d{7}-\d\b"),     # CNIC
    re.compile(r"\b\d{13,}\b"),            # long account/reference numbers
]


def _scrub(text: str) -> str:
    for p in _PII_PATTERNS:
        text = p.sub("[REDACTED]", text)
    return text.strip()


def _numbers_in(text: str) -> List[str]:
    out = []
    for tok in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or ""):
        t = tok.replace(",", "").lstrip("-")
        if not t:
            continue
        try:
            v = float(t)
        except ValueError:
            continue
        out.append(f"{v:.2f}")
    return out


class QuestionEngine:
    """Answers arbitrary customer questions grounded in extracted bill facts and internal evidence."""

    def __init__(self, client: Optional[GeminiModelClient] = None):
        self.client = client or GeminiModelClient()
        self.diagnostics: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ context
    @staticmethod
    def _evidence_view(evidence: InternalEvidence) -> Dict[str, Any]:
        ev = evidence.model_dump(exclude_none=True, exclude={"section_boxes", "source_notes", "bill_id"})
        ev["consumption_history"] = [
            {k: v for k, v in h.items() if k in ("month", "units", "bill_amount", "payment_amount", "pay_date", "status")
             and v is not None}
            for h in ev.get("consumption_history", [])
        ]
        hd = ev.pop("history_diagnostics", {}) or {}
        ev["history_quality"] = {k: hd.get(k) for k in ("record_count", "first_month", "last_month",
                                                        "missing_months_in_range") if k in hd}
        return ev

    def _build_prompt(self, questions: List[Tuple[str, str]], level1: Level1Output,
                      evidence: InternalEvidence) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        l1 = level1.to_level1_dict()
        history = [h.model_dump() for h in evidence.consumption_history]
        facts = calc.build_fact_sheet(l1, history, level1.bill_month, [q for _, q in questions])
        ev = self._evidence_view(evidence)
        prompt = QA_PROMPT.format(
            provider=level1.provider or "unknown provider",
            bill_month_name=calc.month_name(level1.bill_month) if level1.bill_month else "unknown",
            level1=json.dumps(l1, ensure_ascii=False),
            evidence=json.dumps(ev, ensure_ascii=False),
            facts=json.dumps(facts, ensure_ascii=False),
            questions="\n".join(f"- [{qid}] {q}" for qid, q in questions),
        )
        return prompt, facts, {"level1": l1, "evidence": ev}

    @staticmethod
    def _image_bytes(image_path: Optional[str]) -> Optional[bytes]:
        if not image_path:
            return None
        try:
            from vision_utils import load_image, optimize_image_for_llm, image_to_bytes
            return image_to_bytes(optimize_image_for_llm(load_image(image_path), 1600), format="JPEG", quality=90)
        except Exception:
            return None

    def _call(self, prompt: str, image_bytes: Optional[bytes], expected_qids: Optional[List[str]] = None) -> Dict[str, str]:
        raw = self.client.generate(prompt=prompt, image_bytes=image_bytes, image_mime_type="image/jpeg",
                                   response_mime_type="application/json")
        txt = raw.strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            s, e = txt.find("{"), txt.rfind("}")
            data = json.loads(txt[s:e + 1]) if s >= 0 < e else {}
        out: Dict[str, str] = {}
        items = data.get("answers", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = []

        qids = expected_qids or []

        # 1. Match by exact or partial question_id
        for it in items:
            if isinstance(it, dict) and it.get("question_id") is not None and str(it.get("answer", "")).strip():
                ret_id = str(it["question_id"]).strip()
                ans_text = str(it["answer"]).strip()
                if ret_id in qids:
                    out[ret_id] = ans_text
                else:
                    for exp in qids:
                        if exp not in out and (ret_id == exp or exp.endswith(ret_id) or ret_id.endswith(exp)):
                            out[exp] = ans_text
                            break

        # 2. Positional fallback if counts match and some are missing
        if len(items) == len(qids):
            for it, exp in zip(items, qids):
                if exp not in out and isinstance(it, dict) and str(it.get("answer", "")).strip():
                    out[exp] = str(it["answer"]).strip()

        return out

    # -------------------------------------------------------------------- API
    def answer_batch(self, questions: List[Tuple[str, str]], level1: Level1Output, evidence: InternalEvidence,
                     image_path: Optional[str] = None, bill_id: Optional[str] = None) -> Dict[str, str]:
        """Answer all questions of one bill. questions = [(question_id, question_text), ...]."""
        img = self._image_bytes(image_path)
        prompt, facts, ctx = self._build_prompt(questions, level1, evidence)
        qids = [qid for qid, _ in questions]
        answers: Dict[str, str] = {}
        try:
            answers = self._call(prompt, img, expected_qids=qids)
        except Exception as e:
            self.diagnostics.append({"bill_id": bill_id, "batch_error": str(e)[:300]})

        # If any question was missed by the batch, try single-question fallback
        for qid, q in questions:
            if qid not in answers:
                try:
                    p1, _, _ = self._build_prompt([(qid, q)], level1, evidence)
                    res = self._call(p1, img, expected_qids=[qid])
                    if res:
                        answers.update(res)
                except Exception as e:
                    self.diagnostics.append({"bill_id": bill_id, "question_id": qid, "error": str(e)[:300]})
            if not answers.get(qid):
                answers[qid] = "The bill does not provide enough readable information to answer this question."

        support = set(_numbers_in(json.dumps(facts) + json.dumps(ctx)))
        for qid, q in questions:
            answers[qid] = _scrub(answers[qid])
            allowed = support | set(_numbers_in(q))
            unsupported = [n for n in _numbers_in(answers[qid])
                           if n not in allowed and float(n) > 31 and not (1900 <= float(n) <= 2100)]
            self.diagnostics.append({"bill_id": bill_id, "question_id": qid, "unsupported_numbers": unsupported})
        return answers

    def answer_question(self, question: str, level1: Level1Output, evidence: InternalEvidence,
                        bill_id: Optional[str] = None, image_path: Optional[str] = None) -> str:
        """Single-question convenience wrapper (used by the Streamlit demo)."""
        return self.answer_batch([("q", question)], level1, evidence, image_path=image_path, bill_id=bill_id)["q"]
