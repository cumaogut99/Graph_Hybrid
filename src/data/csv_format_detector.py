#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV/TXT Format Algılama
=======================

Ölçüm cihazı export'ları genellikle gerçek kolon başlığından önce açıklama
satırları (Machine:, Operator:, ...) ve başlıktan sonra birim satırı
([s];[g];[degC]) içerir. Bu modül dosyanın ilk satırlarına bakarak şunları
tahmin eder:

- Ayırıcı (',', ';', tab, '|', boşluk)
- Header satırı (0-indexed, fiziksel satır numarası)
- Veri başlangıç satırı (0-indexed, fiziksel satır numarası)
- Ondalık ayırıcının virgül olup olmadığı (örn. 0,25)

Satır numaraları, import dialog'u ve CsvToMpaiConverter ile aynı şekilde
dosyadaki fiziksel satırları (boş satırlar dahil) sayar.
"""

import csv
import re
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Öncelik sırası: eşit skorda listede önce gelen seçilir
CANDIDATE_DELIMITERS = [',', ';', '\t', '|', ' ']

_NUMBER_DOT = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
_NUMBER_COMMA = re.compile(r'^[+-]?(\d+,?\d*|,\d+)([eE][+-]?\d+)?$')
# Tarih/saat benzeri değerler (2024-01-15 14:30:45, 15/01/2024, 14:30:45.123)
_DATETIME_LIKE = re.compile(r'^\d{1,4}[-/.:]\d{1,2}([-/.:T ]\d{1,4})*([.,]\d+)?$')


@dataclass
class CsvFormat:
    delimiter: str
    header_row: Optional[int]  # None = header yok
    start_row: int
    decimal_comma: bool
    column_count: int


def _split(line: str, delimiter: str) -> List[str]:
    if delimiter == ' ':
        return line.split()
    try:
        fields = next(csv.reader([line], delimiter=delimiter))
    except (csv.Error, StopIteration):
        fields = line.split(delimiter)
    return [f.strip().strip('"') for f in fields]


def _is_blank(line: str, delimiter: str) -> bool:
    """Boş satır veya sadece ayırıcıdan oluşan satır (Excel'deki boş satır: ',,')."""
    return not line.strip() or not any(_split(line, delimiter))


def _is_value(field: str, decimal_comma: bool) -> bool:
    """Alan sayısal veya tarih/saat değeri mi?"""
    if not field:
        return False
    pattern = _NUMBER_COMMA if decimal_comma else _NUMBER_DOT
    return bool(pattern.match(field) or _DATETIME_LIKE.match(field))


def _value_ratio(fields: List[str], decimal_comma: bool) -> float:
    """Boş olmayan alanların ne kadarı sayı/tarih değeri."""
    non_empty = [f for f in fields if f]
    if not non_empty:
        return 0.0
    return sum(1 for f in non_empty if _is_value(f, decimal_comma)) / len(non_empty)


def _is_data_line(fields: List[str], decimal_comma: bool) -> bool:
    """Alanların çoğu değer içeriyorsa satır veri satırıdır."""
    return _value_ratio(fields, decimal_comma) >= 0.5


def _looks_like_units(fields: List[str]) -> bool:
    """[s];[g];[degC] veya (s),(g) gibi birim satırı mı?"""
    non_empty = [f for f in fields if f]
    return bool(non_empty) and all(
        (f[0] in '[(' and f[-1] in '])') for f in non_empty
    )


def _analyze(lines: List[str], delimiter: str, decimal_comma: bool):
    """
    Verilen ayırıcı için dosya sonundan geriye doğru tutarlı veri bloğunu bul.

    Returns:
        (skor, kolon_sayısı, veri_başlangıç_satırı) veya None.
        Skor bir tuple'dır: önce veri satırı sayısı, sonra alanların ne kadar
        temiz sayıya çevrildiği, en son kolon sayısı karşılaştırılır. Böylece
        "0,00;0,20" satırını virgülle bölmek (yarısı sayı) noktalı virgülle
        bölmeye (hepsi sayı) tercih edilmez.
    """
    split_lines = [None if _is_blank(l, delimiter) else _split(l, delimiter) for l in lines]

    # Örneklemin sonundaki veri satırlarından kolon sayısını belirle
    tail_counts = [len(f) for f in split_lines[-20:] if f is not None]
    if not tail_counts:
        return None
    column_count = max(set(tail_counts), key=tail_counts.count)
    if column_count < 2:
        return None

    # Sondan geriye: aynı kolon sayısına sahip veri satırlarından oluşan blok
    start = None
    data_rows = 0
    ratio_sum = 0.0
    for i in range(len(split_lines) - 1, -1, -1):
        fields = split_lines[i]
        if fields is None:
            continue  # boş satırlar bloğu bölmez
        ratio = _value_ratio(fields, decimal_comma)
        if len(fields) == column_count and ratio >= 0.5:
            start = i
            data_rows += 1
            ratio_sum += ratio
        else:
            break

    if start is None or data_rows == 0:
        return None
    score = (data_rows, round(ratio_sum / data_rows, 3), column_count)
    return score, column_count, start


def detect_csv_format(file_path: str, encoding: str = 'utf-8',
                      max_lines: int = 500, max_bytes: int = 1_000_000) -> Optional[CsvFormat]:
    """
    Dosyanın başından örnek alarak formatı tahmin et.

    Returns:
        CsvFormat veya algılanamazsa None
    """
    lines: List[str] = []
    read_bytes = 0
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        for line in f:
            line = line.rstrip('\r\n')
            # Tüm satırı saran tırnakları kaldır (Excel export hatası)
            if line.startswith('"') and line.endswith('"') and len(line) > 1 and line.count('"') == 2:
                line = line[1:-1]
            lines.append(line)
            read_bytes += len(line) + 1
            if len(lines) >= max_lines or read_bytes >= max_bytes:
                break

    # Dosya örneklem sınırında kesildiyse son satır yarım olabilir
    if len(lines) >= max_lines or read_bytes >= max_bytes:
        lines = lines[:-1]

    best = None  # (skor, delimiter, decimal_comma, column_count, start)
    for delimiter in CANDIDATE_DELIMITERS:
        # Virgül hem ayırıcı hem ondalık olamaz
        for decimal_comma in ((False,) if delimiter == ',' else (False, True)):
            result = _analyze(lines, delimiter, decimal_comma)
            if result is None:
                continue
            score, column_count, start = result
            if best is None or score > best[0]:
                best = (score, delimiter, decimal_comma, column_count, start)

    if best is None:
        logger.info(f"[FORMAT DETECT] No consistent data block found in {file_path}")
        return None

    _, delimiter, decimal_comma, column_count, start = best

    # Header: veri bloğundan önce, aynı kolon sayısına sahip, değer içermeyen
    # satırlardan en çok dolu alana sahip olan; eşitlikte en üstteki. Excel'den
    # gelen açıklama satırları boş hücrelerle doldurulur ("Machine: A,,"), bu
    # yüzden sadece kolon sayısına bakmak yetmez. Birim satırları atlanır.
    header_row = None
    header_filled = 0
    units_fallback = None
    for i in range(start):
        if _is_blank(lines[i], delimiter):
            continue
        fields = _split(lines[i], delimiter)
        if len(fields) != column_count or _is_data_line(fields, decimal_comma):
            continue
        if _looks_like_units(fields):
            if units_fallback is None:
                units_fallback = i
            continue
        filled = sum(1 for f in fields if f)
        if filled > header_filled:
            header_row, header_filled = i, filled
    if header_row is None:
        header_row = units_fallback

    fmt = CsvFormat(
        delimiter=delimiter,
        header_row=header_row,
        start_row=start,
        decimal_comma=decimal_comma,
        column_count=column_count,
    )
    logger.info(f"[FORMAT DETECT] {file_path}: {fmt}")
    return fmt
