"""Data, constants and helpers shared by the chapters of the cohort-means book.

Each chapter runs in its own kernel and starts with `from cohort_common import *`. The slow steps (per-breast and per-scan
Gaussian fits) are cached in _cache/ and recomputed only when this file or the scan store changes.
"""
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import binomtest

BOOK_DIR = Path(__file__).resolve().parent
DATA_DIR = BOOK_DIR.parent        # Data/: csv and xlsx inputs
CODE_DIR = DATA_DIR.parent        # nova_scan_explorer.py
IMAGES_DIR = CODE_DIR / "Images"  # the pathology and radiology sheets
CACHE_DIR = BOOK_DIR / "_cache"
sys.path.insert(0, str(CODE_DIR))
import nova_scan_explorer as ne


def cached(name, compute, *sources):
    """compute(), stored in _cache/<name>.pkl and reused until this file or any of sources is modified."""
    stamp = [Path(__file__).stat().st_mtime] + [Path(s).stat().st_mtime for s in sources]
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            saved_stamp, value = pickle.load(f)
        if saved_stamp == stamp:
            return value
    value = compute()
    CACHE_DIR.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((stamp, value), f)
    return value


# ---- the prepared data ----
COHORTS = ("NORMAL", "BENIGN", "CANCER")
MAP_COHORTS = ("BENIGN", "CANCER")  # the heatmaps show non-normal samples only
NOVA_RANGE = (250, np.inf)
WATER = (20.0, 22.0)  # nm⁻¹: a sample's water peak height is the mean of its prepared curve over this range

scans, q = ne.read_scans(ne.STORE_PATH)
# the prepared rows, one per sample, in the explorer's order
by_scan = ne.cohort_view(ne.scan_view(scans, q, "stretch"), MAP_COHORTS)
novas, rows, names, _, measures = ne.sample_view(by_scan)
# every patient with a NORMAL, BENIGN or CANCER sample, for the breast comparisons, which are not limited to the heatmap rows
every_novas = ne.sample_view(ne.cohort_view(ne.scan_view(scans, q, "stretch"), COHORTS))[0]
patient_novas = sorted({int(n) for n in every_novas if NOVA_RANGE[0] <= n <= NOVA_RANGE[1]})
kept = [k for k, nova in enumerate(novas) if NOVA_RANGE[0] <= nova <= NOVA_RANGE[1]]
rows, names, measures = rows[kept], [names[k] for k in kept], measures[kept]
# sample_view sorts by Nova, then side, then sample; reversed, the smallest Nova is at the bottom
rows, names = rows[::-1], names[::-1]
with np.errstate(all="ignore"):
    water = np.nanmean(rows[:, (q >= WATER[0]) & (q <= WATER[1])], axis=1)
by_water = np.argsort(-water, kind="stable")  # highest water peak at the top
by_label = ne.scan_view(scans, q, "stretch")

cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("lightgray")

# ---- Water:Fat: two Gaussians ----
FIT_RANGE = (7.0, 25.0)  # nm⁻¹: the two Gaussians are fitted over this range
FAT_COLOUR, WATER_COLOUR = "#d4a017", "tab:blue"


def gaussian(x, height, centre, sigma):
    return height * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def two_gaussians(x, h1, c1, s1, h2, c2, s2):
    return gaussian(x, h1, c1, s1) + gaussian(x, h2, c2, s2)


def area_3sigma(height, centre, sigma):
    return height * sigma * np.sqrt(2 * np.pi) * erf(3 / np.sqrt(2))


def fit_two_gaussians(curve):
    """(fat height, centre, sigma, water height, centre, sigma) fitted to curve over FIT_RANGE, or None."""
    inside = (q >= FIT_RANGE[0]) & (q <= FIT_RANGE[1]) & np.isfinite(curve)
    if inside.sum() < 50 or q[inside].max() < 23:
        return None  # older scans stop before the water peak
    try:
        p, _ = curve_fit(two_gaussians, q[inside], curve[inside], p0=[1, 14, 1.5, 0.5, 21, 3],
                         bounds=([0, 12, 0.5, 0, 17, 0.5], [3, 16, 5, 3, 26, 10]), maxfev=20000)
    except RuntimeError:
        return None
    return p


def water_fat(curve):
    p = fit_two_gaussians(curve)
    return np.nan if p is None else area_3sigma(*p[3:]) / area_3sigma(*p[:3])


def breast_curves(nova):
    """{side: (label, the mean of that breast's prepared scans on q, how many scans)} for a breast whose scans all have one label."""
    by_side = {}
    for side, label, scan_q, curve, _ in ne.read_patients(ne.STORE_PATH, {nova}, ne.REFERENCES["stretch"])[nova][0]:
        by_side.setdefault(side, ([], set()))[0].append(np.interp(q, scan_q, curve, left=np.nan, right=np.nan))
        by_side[side][1].add(label)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {side: (labels.pop(), np.nanmean(curves, axis=0), len(curves))
                for side, (curves, labels) in sorted(by_side.items()) if len(labels) == 1}


# ---- patients with a large left/right intensity imbalance ----
INTENSITY_RATIO_MAX = 2.0  # drop a patient if max(LEFT, RIGHT) / min(LEFT, RIGHT) of mean image totals is ≥ this
totals = pd.read_csv(DATA_DIR / "breast_image_totals.csv")
intensity_ratio = {}  # nova: brighter / dimmer mean image total
for nova, group in totals.groupby("nova"):
    sides = {row.side: float(row.image_total_mean) for row in group.itertuples()}
    if "LEFT" in sides and "RIGHT" in sides and min(sides.values()) > 0:
        intensity_ratio[int(nova)] = max(sides.values()) / min(sides.values())
intensity_skip = {nova for nova, ratio in intensity_ratio.items() if ratio >= INTENSITY_RATIO_MAX}
analysis_novas = [nova for nova in patient_novas if nova not in intensity_skip]

# ---- Water:Fat in every patient ----
# name: (the breast expected to have less water, the labels of the breast expected to have more)
COMPARISONS = {"CANCER vs NORMAL": ("NORMAL", ("CANCER",)), "BENIGN vs NORMAL": ("NORMAL", ("BENIGN",)),
               "CANCER vs BENIGN": ("BENIGN", ("CANCER",))}
all_novas = analysis_novas


def _every_breast():
    found = {}  # (nova, side): (label, Water:Fat)
    for nova in all_novas:
        for side, (label, curve, _) in breast_curves(nova).items():
            found[nova, side] = (label, water_fat(curve))
    return found


# Water:Fat of every breast, from all of its patient's scans (intensity-imbalanced patients already removed)
breast = cached("breast_water_fat", _every_breast, ne.STORE_PATH, DATA_DIR / "breast_image_totals.csv")


def pairs(novas_here, base, others):
    """(base breast's Water:Fat, other breast's) for patients in novas_here with one `base` breast and one breast labelled any of `others`."""
    found = []
    for nova in novas_here:
        sides = [breast.get((nova, side)) for side in ("LEFT", "RIGHT")]
        if None in sides or not all(np.isfinite(ratio) for _, ratio in sides):
            continue
        (label_a, ratio_a), (label_b, ratio_b) = sides
        if label_a == base and label_b in others:
            found.append((ratio_a, ratio_b))
        elif label_b == base and label_a in others:
            found.append((ratio_b, ratio_a))
    return found


# the water-rich top and the fatty bottom PART% of the rows of Figure 2 (right)
PART = 15  # % of the rows
part_size = round(len(names) * PART / 100)
TOP_COLOUR, BOTTOM_COLOUR = "tab:blue", "#d4a017"


def top_and_bottom(top_novas, bottom_novas):
    """{"top": patients, "bottom": patients}, with any patient in the top group left out of the bottom one, so no patient is counted in both."""
    return {"top": set(top_novas), "bottom": set(bottom_novas) - set(top_novas)}


# ---- the on-the-fly rule ----
WATER_MIN = 0.35    # a usable scan's water peak (mean over q 20-22, stretched) reaches 35% of the fat peak
CENTRE_MIN = 19.0   # nm⁻¹: and its fitted water Gaussian is centred at q >= 19, a real water peak


def _every_scan():
    patient_scans = ne.read_patients(ne.STORE_PATH, set(analysis_novas), ne.REFERENCES["stretch"])
    records = []  # (nova, side, label, curve on q, water peak height, fitted water centre)
    for nova in analysis_novas:
        for side, label, scan_q, curve, _ in patient_scans[nova][0]:
            on_q = np.interp(q, scan_q, curve, left=np.nan, right=np.nan)
            with np.errstate(all="ignore"):
                height = np.nanmean(on_q[(q >= WATER[0]) & (q <= WATER[1])])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p = fit_two_gaussians(on_q)
            records.append((nova, side, label, on_q, height, p[4] if p is not None else np.nan))
    return records


# every scan of every patient from Nova 250 onwards, fitted on its own
scan_records = cached("scan_records", _every_scan, ne.STORE_PATH, DATA_DIR / "breast_image_totals.csv")


def usable(height, centre, min_height, min_centre):
    return bool(np.isfinite(height) and np.isfinite(centre) and height >= min_height and centre >= min_centre)


def rule_breasts(min_height, min_centre, novas=None, relabel=None, skip=(), rejected=False):
    """{(nova, side): (label, Water:Fat of the mean of its usable scans, usable scans, all scans)} for breasts with a usable scan and one label.
    min_height is a number or a function of the Nova; relabel {(nova, side): label} overrides stored labels; patients in skip are left out;
    rejected=True keeps the scans the rule rejects instead."""
    kept, counts = defaultdict(list), defaultdict(int)
    for nova, side, label, curve, height, centre in scan_records:
        if (novas is not None and nova not in novas) or nova in skip:
            continue
        label = (relabel or {}).get((nova, side), label)
        counts[nova, side] += 1
        threshold = min_height(nova) if callable(min_height) else min_height
        if usable(height, centre, threshold, min_centre) != rejected:
            kept[nova, side].append((label, curve))
    out = {}
    for key, items in kept.items():
        labels = {label for label, _ in items}
        if len(labels) == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[key] = (labels.pop(), water_fat(np.nanmean([c for _, c in items], axis=0)), len(items), counts[key])
    return out


def rule_results(min_height, min_centre, novas=None, relabel=None, skip=(), rejected=False):
    """{comparison: (first-named breast higher, patients compared, p-value)} for patients with both breasts usable."""
    found = rule_breasts(min_height, min_centre, novas, relabel, skip, rejected)
    both = sorted({n for n, _ in found if (n, "LEFT") in found and (n, "RIGHT") in found})
    results = {}
    for comparison, (base, others) in COMPARISONS.items():
        higher = compared = 0
        for nova in both:
            (label_a, ratio_a, *_), (label_b, ratio_b, *_) = found[nova, "LEFT"], found[nova, "RIGHT"]
            if not (np.isfinite(ratio_a) and np.isfinite(ratio_b)):
                continue
            if label_a == base and label_b in others:
                compared += 1; higher += ratio_b > ratio_a
            elif label_b == base and label_a in others:
                compared += 1; higher += ratio_a > ratio_b
        results[comparison] = (int(higher), compared, binomtest(int(higher), compared).pvalue if compared else np.nan)
    return results


def rule_text(results, comparison):
    higher, total, _ = results[comparison]
    return f"{higher}/{total} ({higher / total:.0%})" if total else "no patients"


# each breast's usable and rejected scans; a breast can have both kinds
breast_rej, breast_use = defaultdict(list), defaultdict(list)  # (nova, side, label): [curves]
for nova, side, label, curve, height, centre in scan_records:
    if label not in COHORTS:
        continue
    if usable(height, centre, WATER_MIN, CENTRE_MIN):
        breast_use[nova, side, label].append(curve)
    else:
        breast_rej[nova, side, label].append(curve)

USABLE_COLOUR, REJECTED_COLOUR = TOP_COLOUR, "grey"

# age is stored per scan in the HDF5; one value per Nova
nova_age = {}
with h5py.File(ne.STORE_PATH, "r") as _store:
    for _group in _store["scans"].values():
        _age = float(_group.attrs.get("age", np.nan))
        if np.isfinite(_age):
            nova_age[int(_group.attrs["nova"])] = _age


def box_with_points(ax, groups, rng):
    """groups: [(x position, values, colour)]; a box plot of each with every value as a jittered dot."""
    for x, values, colour in groups:
        if not len(values):
            continue
        ax.boxplot([values], positions=[x], widths=0.32, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=colour, alpha=0.35, edgecolor=colour), medianprops=dict(color="#1f2d4a", lw=2),
                   whiskerprops=dict(color=colour), capprops=dict(color=colour))
        ax.scatter(x + rng.uniform(-0.1, 0.1, len(values)), values, s=9, color="#3b4a63", alpha=0.7, lw=0, zorder=3)
