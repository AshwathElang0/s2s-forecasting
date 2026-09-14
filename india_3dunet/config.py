"""Shared constants for the India 3D U-Net experiment.

Grid, lead-time, and split choices are pinned to literature-review.md
sections 1, 5.1 and 7.1 (one level up, in the repo root) -- see plan.md
for the reasoning behind each one.
"""
from pathlib import Path

# Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT.parent / "data"  # ../data relative to this folder, i.e. s2s-forecasting/data
ECMWF_ROOT = DATA_ROOT / "Model data ecmwf"
IMD_RAIN_FILE = DATA_ROOT / "IMD Rainfall" / "merged_imd_rain_data_1951_2025.nc"
IMD_TMAX_FILE = DATA_ROOT / "IMD Temperature" / "merged_imd_tmax_data_1951_2025.nc"
IMD_TMIN_FILE = DATA_ROOT / "IMD Temperature" / "merged_imd_tmin_data_1951_2025.nc"
CACHE_ROOT = REPO_ROOT / "data_cache"

# ECMWF native grid (data.md) -------------------------------------------
NATIVE_LAT = 40
NATIVE_LON = 60
NATIVE_LAT_VALS = (39.0, -19.5)   # (first, last) -- descending
NATIVE_LON_VALS = (21.0, 109.5)   # (first, last) -- ascending
GRID_RES_DEG = 1.5

# Padded grid fed to the 3D U-Net so 3 rounds of 2x pooling land cleanly
# (48 -> 24 -> 12 -> 6, 64 -> 32 -> 16 -> 8), per literature-review.md S5.2.
PAD_LAT = 48
PAD_LON = 64
PAD_TOP = 4      # rows of padding added before latitude index 0
PAD_LEFT = 2     # cols of padding added before longitude index 0

# Lead time: 46 raw daily steps are available; we use the first 32
# (weeks 1-4 plus 4 spare days) so the lead axis also pools cleanly
# 3 times (32 -> 16 -> 8 -> 4), matching the reference paper's own lead
# dimension exactly.
LEAD_DAYS = 32
TARGET_WEEKS = 4  # the days actually scored -- days 1..28

# Ensemble ----------------------------------------------------------------
N_MEMBERS = 11  # 1 control (_cf_) + 10 perturbed (_pf_)

# ECMWF variables and their pressure levels (data.md) ---------------------
ECMWF_VARS = {
    "tp": {"var": "tp", "levels": None},
    "wtmp": {"var": "sst", "levels": None},
    "msl": {"var": "msl", "levels": None},
    "gh": {"var": "gh", "levels": [200, 1000]},
    "q": {"var": "q", "levels": [200, 500, 700, 850]},
    # NB: ../s2s-forecasting/data/data.md's table lists u/v levels as
    # "200, 1000 mb" -- verified against the actual .nc files and that's
    # wrong; the real levels are 200/850 (gh is the one that's 200/1000).
    "u": {"var": "u", "levels": [200, 850]},
    "v": {"var": "v", "levels": [200, 850]},
}

# Targets -------------------------------------------------------------
TARGETS = ["rain", "tmax", "tmin"]

# Reforecast issue dates span 2004-01-04 .. 2023-12-27 (weekly-ish, ~1039
# dates, literature-review.md S1.1). Fixed calendar blocks for the first
# pass; blocked CV averaging is the literature review's S7.1 upgrade.
TRAIN_YEARS = range(2004, 2018)   # 14 years
VAL_YEARS = range(2018, 2021)     # 3 years
TEST_YEARS = range(2021, 2024)    # 3 years

RANDOM_SEED = 0
