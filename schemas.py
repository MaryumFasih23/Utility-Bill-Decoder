"""
schemas.py - Pydantic models and schemas for WTQ 2026 Utility Bill Decoder.
Strict adherence to competition rules for Level 1 JSON and Internal Evidence for Level 2.
"""

from typing import List, Optional, Literal, Dict, Any, Union
import re
from pydantic import BaseModel, Field, field_validator, model_validator


# Allowed literal types
ProviderType = Literal["KE", "LESCO", "IESCO"]
ChargeType = Literal[
    "energy",
    "fixed",
    "fpa",
    "quarterly_adjustment",
    "surcharge",
    "meter_rent",
    "subsidy",
    "other"
]
TaxType = Literal[
    "gst",
    "electricity_duty",
    "income_tax",
    "municipal_tax",
    "other_tax"
]

_NULL_TOKENS = {
    "", "null", "none", "n/a", "na", "-", "--", "nil", "masked", "unreadable",
    "illegible", "blank", "not printed", "not available",
}

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_amount(v: Any) -> Optional[float]:
    """
    Parse a monetary/numeric value exactly as printed on a bill.

    General rules (provider independent):
    - "54770CR" / "54,770 Cr" / "(54770)" / "54770-" / "-54770" -> negative credit.
    - "39411 / 2" (arrears with an age suffix) -> 39411 (amount before the slash).
    - Thousands separators, "Rs.", "PKR" are stripped.
    - Text with no digits (e.g. "NOT TO BE PAID", "masked") -> None.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = str(v).strip()
    if s.lower() in _NULL_TOKENS:
        return None
    s = s.replace("\u2212", "-").replace("\u2013", "-")
    s = re.sub(r"(?i)\b(rs\.?|pkr)\s*", "", s).strip()
    s = s.replace(",", "").replace(" ", "")
    if not re.search(r"\d", s):
        return None

    negative = False
    if re.search(r"(?i)cr\.?$", s):
        negative = True
        s = re.sub(r"(?i)cr\.?$", "", s)
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    if s.endswith("-"):
        negative = True
        s = s[:-1]
    if s.startswith("-"):
        negative = True
        s = s.lstrip("-")
    s = s.lstrip("+")

    # "39411/2" -> amount before slash (LESCO/IESCO arrears age notation)
    m = re.match(r"^(\d+(?:\.\d+)?)/\d{1,2}$", s)
    if m:
        s = m.group(1)

    m = re.match(r"^\d+(?:\.\d+)?$", s)
    if not m:
        return None
    val = float(s)
    return round(-val if negative else val, 2)


class ChargeItem(BaseModel):
    type: ChargeType
    amount: float

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, v: Any) -> float:
        val = parse_amount(v)
        if val is None:
            raise ValueError(f"Invalid numeric amount: {v}")
        return val


class TaxItem(BaseModel):
    type: TaxType
    amount: float

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, v: Any) -> float:
        val = parse_amount(v)
        if val is None:
            raise ValueError(f"Invalid numeric amount: {v}")
        return val


def _normalize_date(v: Optional[str]) -> Optional[str]:
    if not v or not isinstance(v, str):
        return None
    v = v.strip()
    if v.lower() in _NULL_TOKENS:
        return None
    # YYYY-MM-DD
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", v)
    if m:
        year, month, day = m.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    # DD-MM-YYYY or DD/MM/YYYY
    m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", v)
    if m:
        day, month, year = m.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    # DD-Mon-YYYY / DD Mon YY / 21st Apr 2026
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?[-/\s]*([a-zA-Z]{3,9})[-/\s,]*(\d{2,4})$", v)
    if m:
        day, mon_str, year_str = m.groups()
        mon = _MONTH_MAP.get(mon_str.lower()[:3])
        if mon:
            year = int(year_str)
            if year < 100:
                year += 2000
            return f"{year:04d}-{mon:02d}-{int(day):02d}"
    return None


def _normalize_month(v: Optional[str]) -> Optional[str]:
    """Normalize a printed month label (with year) to YYYY-MM. Returns None if no year."""
    if v is None:
        return None
    v = str(v).strip()
    if v.lower() in _NULL_TOKENS:
        return None
    # YYYY-MM
    m = re.match(r"^(\d{4})[-/.](\d{1,2})$", v)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        return f"{year:04d}-{month:02d}" if 1 <= month <= 12 else None
    # MM/YY or MM/YYYY (KE billing history "03/26")
    m = re.match(r"^(\d{1,2})[-/.](\d{2}|\d{4})$", v)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if year < 100:
            year += 2000
        return f"{year:04d}-{month:02d}" if 1 <= month <= 12 else None
    # Mon-YY / Mon YYYY / Sep25 / September 2025
    m = re.match(r"^([a-zA-Z]{3,9})\.?[-/\s']*(\d{2}|\d{4})$", v)
    if m:
        mon = _MONTH_MAP.get(m.group(1).lower()[:3])
        if mon:
            year = int(m.group(2))
            if year < 100:
                year += 2000
            return f"{year:04d}-{mon:02d}"
    # YYYY-Mon
    m = re.match(r"^(\d{4})[-/\s]*([a-zA-Z]{3,9})$", v)
    if m:
        mon = _MONTH_MAP.get(m.group(2).lower()[:3])
        if mon:
            return f"{int(m.group(1)):04d}-{mon:02d}"
    return None


def month_label_to_number(label: Any) -> Optional[int]:
    """'Apr' / 'APRIL' / '04' -> 4 (month without year)."""
    if label is None:
        return None
    s = str(label).strip()
    m = re.match(r"^([a-zA-Z]{3,9})", s)
    if m:
        return _MONTH_MAP.get(m.group(1).lower()[:3])
    m = re.match(r"^(\d{1,2})$", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(1))
    return None


def _filter_items(v: Any) -> List[Dict[str, Any]]:
    """Drop blank/zero/unparseable line items; keep repeated lines separate."""
    if not isinstance(v, list):
        return []
    cleaned = []
    for item in v:
        if isinstance(item, BaseModel):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        amt = parse_amount(item.get("amount"))
        if amt is None or amt == 0:
            continue
        cleaned.append({"type": item.get("type"), "amount": amt})
    return cleaned


class Level1Output(BaseModel):
    """
    Official Level 1 extraction schema.
    Must output exact keys in exact order with canonical types.
    """
    provider: Optional[ProviderType] = None
    tariff: Optional[str] = None
    sanctioned_load_kw: Optional[float] = None
    bill_month: Optional[str] = None
    reading_date: Optional[str] = None
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    previous_reading: Optional[float] = None
    current_reading: Optional[float] = None
    units_consumed: Optional[float] = None
    charges: List[ChargeItem] = Field(default_factory=list)
    total_charges: Optional[float] = None
    taxes: List[TaxItem] = Field(default_factory=list)
    total_taxes: Optional[float] = None
    current_bill: Optional[float] = None
    arrears: Optional[float] = None
    payable_within_due_date: Optional[float] = None
    payable_after_due_date: Optional[float] = None

    @field_validator("provider", mode="before")
    @classmethod
    def clean_provider(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().upper()
        if s in ("KE", "K-ELECTRIC", "KESC", "K ELECTRIC"):
            return "KE"
        if "LESCO" in s or "LAHORE" in s:
            return "LESCO"
        if "IESCO" in s or "ISLAMABAD" in s:
            return "IESCO"
        return None

    @field_validator("tariff", mode="before")
    @classmethod
    def clean_tariff(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = re.sub(r"\s+", " ", str(v)).strip()
        return None if s.lower() in _NULL_TOKENS else s

    @field_validator("bill_month", mode="before")
    @classmethod
    def clean_bill_month(cls, v: Any) -> Optional[str]:
        return _normalize_month(str(v) if v is not None else None)

    @field_validator("reading_date", "issue_date", "due_date", mode="before")
    @classmethod
    def clean_dates(cls, v: Any) -> Optional[str]:
        return _normalize_date(str(v) if v is not None else None)

    @field_validator(
        "sanctioned_load_kw",
        "previous_reading",
        "current_reading",
        "units_consumed",
        "total_charges",
        "total_taxes",
        "current_bill",
        "arrears",
        "payable_within_due_date",
        "payable_after_due_date",
        mode="before"
    )
    @classmethod
    def clean_numeric(cls, v: Any) -> Optional[float]:
        return parse_amount(v)

    @field_validator("charges", "taxes", mode="before")
    @classmethod
    def filter_items(cls, v: Any) -> List[Dict[str, Any]]:
        return _filter_items(v)

    def to_level1_dict(self) -> Dict[str, Any]:
        """Produce exact dictionary conforming to Level 1 schema order and rules."""
        def format_num(n: Optional[float]) -> Optional[Union[int, float]]:
            if n is None:
                return None
            if float(n).is_integer():
                return int(n)
            return round(float(n), 2)

        return {
            "provider": self.provider,
            "tariff": self.tariff,
            "sanctioned_load_kw": format_num(self.sanctioned_load_kw),
            "bill_month": self.bill_month,
            "reading_date": self.reading_date,
            "issue_date": self.issue_date,
            "due_date": self.due_date,
            "previous_reading": format_num(self.previous_reading),
            "current_reading": format_num(self.current_reading),
            "units_consumed": format_num(self.units_consumed),
            "charges": [
                {"type": c.type, "amount": format_num(c.amount)}
                for c in self.charges
            ],
            "total_charges": format_num(self.total_charges),
            "taxes": [
                {"type": t.type, "amount": format_num(t.amount)}
                for t in self.taxes
            ],
            "total_taxes": format_num(self.total_taxes),
            "current_bill": format_num(self.current_bill),
            "arrears": format_num(self.arrears),
            "payable_within_due_date": format_num(self.payable_within_due_date),
            "payable_after_due_date": format_num(self.payable_after_due_date)
        }


# =========================================================================
# Internal Evidence Model for Level 2 Q&A
# =========================================================================

class _NumericCoercion(BaseModel):
    """Coerce any float-typed field from printed strings via parse_amount."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_numbers(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name, field in cls.model_fields.items():
            if name in out and "float" in str(field.annotation) and not isinstance(out[name], (list, dict)):
                out[name] = parse_amount(out[name])
        return out


class MonthlyHistoryEntry(_NumericCoercion):
    month: str  # canonical YYYY-MM
    units: Optional[float] = None
    bill_amount: Optional[float] = None
    payment_amount: Optional[float] = None
    pay_date: Optional[str] = None
    status: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # e.g. ["usage_chart", "billing_table"]


class MeterRegisterEntry(_NumericCoercion):
    register_name: Optional[str] = None  # e.g. "Peak", "Off-Peak", "Import", "Export"
    previous_reading: Optional[float] = None
    current_reading: Optional[float] = None
    units: Optional[float] = None
    mdi_kw: Optional[float] = None


class DetailedChargeEntry(_NumericCoercion):
    description: Optional[str] = None
    amount: Optional[float] = None
    units: Optional[float] = None
    rate: Optional[float] = None
    section: Optional[str] = None


class InternalEvidence(_NumericCoercion):
    """
    Evidence model containing all bill details used ONLY for Level 2 Customer Q&A.
    NEVER exported into Level 1 JSON.
    """
    bill_id: Optional[str] = None
    provider: Optional[str] = None
    bill_month: Optional[str] = None
    sanctioned_load_kw: Optional[float] = None

    # Chronological (oldest -> newest) history, months canonical YYYY-MM
    consumption_history: List[MonthlyHistoryEntry] = Field(default_factory=list)
    history_includes_current_month: Optional[bool] = None
    history_diagnostics: Dict[str, Any] = Field(default_factory=dict)

    meter_registers: List[MeterRegisterEntry] = Field(default_factory=list)
    line_item_details: List[DetailedChargeEntry] = Field(default_factory=list)

    # Net metering
    net_metering_present: bool = False
    net_metering_units_imported: Optional[float] = None
    net_metering_units_exported: Optional[float] = None
    net_metering_net_units: Optional[float] = None
    net_metering_benefit_pkr: Optional[float] = None
    outstanding_balance_pkr: Optional[float] = None

    # Financial adjustments & credits (as printed)
    subsidies_or_relief_pkr: Optional[float] = None
    fpa_amount_pkr: Optional[float] = None
    late_payment_surcharge_pkr: Optional[float] = None
    payable_after_due_date_printed_text: Optional[str] = None
    payable_within_due_date_printed_text: Optional[str] = None
    notes: List[str] = Field(default_factory=list)

    # Section boxes (0-1000 normalized ymin,xmin,ymax,xmax) for targeted re-reads
    section_boxes: Dict[str, List[float]] = Field(default_factory=dict)
    source_notes: Dict[str, str] = Field(default_factory=dict)
