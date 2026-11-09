"""The newest patients, Nova 588 onwards, every scan scored by the water-peak rule exactly like the main analysis."""
import warnings
from collections import defaultdict

import numpy as np

from cohort_common import ne, q, WATER, WATER_MIN, CENTRE_MIN, fit_two_gaussians, usable, water_fat

FIRST_NEW = 588
# validation starts after the development set; Nova 588 is already in the main analysis (patient_novas), so it is shown with the newest
# patients but left out of the validation counts
VALIDATION_FIRST = 589
# the scans added on 9 Nov 2026 (Nova 596 onwards, measured 2-6 Nov), prepared exactly like the main store by nova_sdd_peak_align.py
NOV09_STORE = ne.STORE_PATH.with_name("nova_sdd_corrected_1d_2026Nov09.h5")
new_scans = {}
for store_path in [ne.STORE_PATH] + ([NOV09_STORE] if NOV09_STORE.exists() else []):
    stored_novas = {int(group.attrs["nova"]) for group in ne.stored_scans(store_path)}
    new_scans |= ne.read_patients(store_path, {n for n in stored_novas if n >= FIRST_NEW and n not in new_scans}, ne.REFERENCES["stretch"])
new_novas = sorted(new_scans)
validation_novas = [n for n in new_novas if n >= VALIDATION_FIRST]

new_records = defaultdict(list)   # (nova, side): [(label, curve on q, height, centre, usable)]
for nova in new_novas:
    for side, label, scan_q, curve, _ in new_scans[nova][0]:
        on_q = np.interp(q, scan_q, curve, left=np.nan, right=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            height = np.nanmean(on_q[(q >= WATER[0]) & (q <= WATER[1])])
            p = fit_two_gaussians(on_q)
        centre = p[4] if p is not None else np.nan
        new_records[nova, side].append((label, on_q, height, centre, usable(height, centre, WATER_MIN, CENTRE_MIN)))


def breast_summary(nova, side):
    """(label, usable scans, all scans, Water:Fat of the mean of the usable scans or nan)."""
    records = new_records.get((nova, side), [])
    good = [curve for _, curve, _, _, ok in records if ok]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio = water_fat(np.nanmean(good, axis=0)) if good else np.nan
    labels = sorted({label for label, *_ in records}) or ["not scanned"]
    return ", ".join(l if l != "—" else "not yet labelled" for l in labels), len(good), len(records), ratio


def outcome(nova):
    usable_sides = sum(breast_summary(nova, side)[1] > 0 for side in ("LEFT", "RIGHT"))
    return {2: "both breasts usable: compared", 1: "one breast usable: not compared", 0: "neither breast usable: not compared"}[usable_sides]


outcomes = {nova: outcome(nova) for nova in new_novas}
n_both = sum(o.startswith("both") for o in outcomes.values())
n_scans = sum(len(v) for v in new_records.values())
n_usable = sum(ok for v in new_records.values() for *_, ok in v)

# validation counts leave out any Nova already in the development set (Nova 588)
val_outcomes = {n: outcomes[n] for n in validation_novas}
n_both_val = sum(o.startswith("both") for o in val_outcomes.values())
n_scans_val = sum(len(new_records[n, s]) for n in validation_novas for s in ("LEFT", "RIGHT") if (n, s) in new_records)
n_usable_val = sum(ok for n in validation_novas for s in ("LEFT", "RIGHT") for *_, ok in new_records.get((n, s), []))

# the validation: CANCER/NORMAL patients among the held-out newest with both breasts usable and labelled
validation = []   # (nova, CANCER Water:Fat, NORMAL Water:Fat)
for nova in validation_novas:
    left, right = breast_summary(nova, "LEFT"), breast_summary(nova, "RIGHT")
    if outcomes[nova].startswith("both") and {left[0], right[0]} == {"CANCER", "NORMAL"}:
        by_label = {left[0]: left[3], right[0]: right[3]}
        validation.append((nova, by_label["CANCER"], by_label["NORMAL"]))
n_labelled = sum(all(breast_summary(nova, side)[0] in ne.LABELS for side in ("LEFT", "RIGHT")) for nova in validation_novas)
overlap_with_development = [n for n in new_novas if n < VALIDATION_FIRST]
