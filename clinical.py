"""BI-RADS density and age of every patient from the pathology and radiology sheets, and which patients the water-peak rule compares."""
import re
from collections import Counter, defaultdict

import numpy as np
import openpyxl

from cohort_common import IMAGES_DIR, COHORTS, WATER_MIN, CENTRE_MIN, scan_records, usable

# which patients the rule compares (both breasts have a usable scan) and which it drops (at least one breast has none)
breast_usable = defaultdict(bool)
for nova, side, label, curve, height, centre in scan_records:
    if label in COHORTS:
        breast_usable[nova, side] |= usable(height, centre, WATER_MIN, CENTRE_MIN)
scanned_both = sorted({n for n, _ in breast_usable if (n, "LEFT") in breast_usable and (n, "RIGHT") in breast_usable})
n_usable_breasts = {n: breast_usable[n, "LEFT"] + breast_usable[n, "RIGHT"] for n in scanned_both}
rule_groups = {"compared": [n for n in scanned_both if n_usable_breasts[n] == 2],
               "left out": [n for n in scanned_both if n_usable_breasts[n] < 2]}
left_out_one = sum(n_usable_breasts[n] == 1 for n in scanned_both)
left_out_both = sum(n_usable_breasts[n] == 0 for n in scanned_both)

# every patient's clinical text, BI-RADS density (most frequent mention) and age, from the two sheets in Images
SHEETS = ((IMAGES_DIR / "Nova_Patient Pathology Reports .xlsx", 1), (IMAGES_DIR / "Nova Mammo-US-MRI Results.xlsx", 2))
DENSITY = (("almost entirely fatty", "A: almost entirely fatty"), ("scattered areas of fibroglandular", "B: scattered fibroglandular"),
           ("heterogeneously dense", "C: heterogeneously dense"), ("extremely dense", "D: extremely dense"))
clinical_text, density_mentions, patient_age = defaultdict(str), defaultdict(list), {}
for path, header_rows in SHEETS:
    sheet_rows = list(openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    header = [str(x) for x in sheet_rows[0]]
    age_column = header.index("Age (Years)") if "Age (Years)" in header else None
    nova = None
    for r in sheet_rows[header_rows:]:
        if r[0]:
            m = re.search(r"(\d+)", str(r[0]))
            nova = int(m.group(1)) if m else None
        if nova is None:
            continue
        text = " ".join(str(x) for x in r if x is not None).lower()
        clinical_text[nova] += " " + text
        for phrase, category in DENSITY:
            if phrase in text:
                density_mentions[nova].append(category)
                break
        try:
            if age_column is not None and r[age_column]:
                patient_age[nova] = float(r[age_column])
        except ValueError:
            pass
        m = re.search(r"(\d{2})[- ]year[- ]old", text)
        if m and nova not in patient_age:
            patient_age[nova] = float(m.group(1))
patient_density = {n: Counter(v).most_common(1)[0][0] for n, v in density_mentions.items()}
DENSITY_ORDER = [c for _, c in DENSITY] + ["not stated"]
density_share = {group: {c: np.mean([patient_density.get(n, "not stated") == c for n in novas_here]) for c in DENSITY_ORDER}
                 for group, novas_here in rule_groups.items()}
dense_share = {group: shares["C: heterogeneously dense"] + shares["D: extremely dense"] for group, shares in density_share.items()}
group_ages = {group: [patient_age[n] for n in novas_here if n in patient_age] for group, novas_here in rule_groups.items()}
