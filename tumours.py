"""Tumour size and grade of every CANCER breast, from the pathology and radiology sheets."""
import re

import numpy as np
import openpyxl

from cohort_common import IMAGES_DIR, WATER_MIN, CENTRE_MIN, ne, scan_records, rule_breasts


def grade_from_text(text):
    """The overall grade, 1-3, from the report's text, or None when it gives none or only a range."""
    text = str(text).lower()
    score = re.search(r"(\d)\s*/\s*9", text)
    if score:
        return {3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3, 9: 3}.get(int(score.group(1)))
    if "nuclear" in text or "at least" in text or re.search(r"\d\s*-\s*\d", text):
        return None
    stated = re.search(r"grad\w*\s*\(?\s*([123])", text) or re.search(r"\(([123])\)", text)
    return int(stated.group(1)) if stated else None


def read_grades(path):
    """{(nova, side): the highest overall grade of that breast's specimens}. A row without a Patient ID continues the patient above it."""
    rows = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True)
    header = next(rows)
    id_column, specimen_column, grade_column = (header.index(name) for name in ("Patient ID", "Specimen", "Overall Grade"))
    grades, nova = {}, None
    for row in rows:
        if row[id_column]:
            nova = int(re.search(r"\d+", str(row[id_column])).group())
        specimen = str(row[specimen_column] or "").lower()
        side = "LEFT" if "left" in specimen else "RIGHT" if "right" in specimen else None
        grade = grade_from_text(row[grade_column]) if row[grade_column] else None
        if side and grade and "lymph node" not in specimen:
            grades[(nova, side)] = max(grade, grades.get((nova, side), 0))
    return grades


def read_imaging_sizes(path):
    """{(nova, side): the largest mass size in mm in that side's radiology findings (mammogram, ultrasound or MRI)}. Only sentences about a
    mass count; non-mass enhancement, calcification spans, cysts, lymph nodes, benign findings and distances from the nipple are left out."""
    rows = list(openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))[2:]
    side_columns = {"LEFT": (4, 10), "RIGHT": (5, 11)}  # Findings Summary and Detailed Findings, left and right
    number = r"\d+(?:\.\d+)?"
    size_text = re.compile(rf"({number})\s*(?:[x×]\s*({number})\s*)?(?:[x×]\s*({number})\s*)?(mm|cm)\b(?!\s*(?:fn|from))", re.I)
    found, nova = {}, None
    for row in rows:
        if row[0]:
            m = re.search(r"\d+", str(row[0]))
            nova = int(m.group()) if m else None
        if nova is None:
            continue
        for side, columns in side_columns.items():
            for column in columns:
                for sentence in re.split(r"(?<=[.;\n])\s+", str(row[column] or "")):
                    lower = sentence.lower()
                    if "mass" not in lower or re.search(r"non-?\s?mass|cyst|lymph node|calcification|benign", lower):
                        continue
                    for m in size_text.finditer(sentence):
                        size = max(float(v) for v in m.groups()[:3] if v) * (10 if m.group(4).lower() == "cm" else 1)
                        if size <= 100:
                            found[nova, side] = max(size, found.get((nova, side), 0))
    return found


sizes = ne.read_tumour_sizes(ne.PATHOLOGY_PATH)
grades = read_grades(ne.PATHOLOGY_PATH)
imaging_sizes = read_imaging_sizes(IMAGES_DIR / "Nova Mammo-US-MRI Results.xlsx")
cancer_breasts = {(nova, side) for nova, side, label, *_ in scan_records if label == "CANCER"}
usable_breasts = {key for key, (label, *_) in rule_breasts(WATER_MIN, CENTRE_MIN).items() if label == "CANCER"}
