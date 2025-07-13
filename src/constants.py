"""Expresiones regulares y parámetros estáticos compartidos."""
from __future__ import annotations

import re

FAQ_HEADER_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ0-9 &'/-]{4,}:?$")
CURRENCY_RE = re.compile(r"(€|eur(?:os?)?)", re.I)
MONEY_RE = re.compile(
    r"(€\s*\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*(?:€|eur(?:os?)?))",
    re.I,
)
PRICE_QUERY_RE = re.compile(
    r"(cu[aá]nt[oa]?|precio|vale|cuesta|coste|importe|tarifa)", re.I)
