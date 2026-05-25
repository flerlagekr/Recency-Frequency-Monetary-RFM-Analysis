# ============================================================
# RFM SNAPSHOTS (DONOR-LEVEL ONLY)
#
# Input:
#   Gift-level donation file (Donations.csv)
#
# Output:
#   One snapshot file with donor-level RFM metrics
#   - One row per donor per snapshot date
#   - Includes only donors who have given at least one gift
#     on or before the snapshot date
#
# Snapshot logic:
#   - Monthly snapshots use month-end dates
#   - One additional snapshot is computed as of today
#   - Is Current = True for the "as of today" snapshot
#
# RFM scoring:
#   - Recency, Frequency, Monetary scored 0–10
#   - Percentile-rank style (Tableau Prep-like)
#   - Scored independently within each As of Date
#
# Note on merge_asof:
#   pandas merge_asof REQUIRES the key columns (left_on/right_on)
#   to be globally sorted ascending. That means sorting by the DATE
#   key first (not by Donor ID first).
# ============================================================

from datetime import datetime, date
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

IN_FILE = r"C:\Ken\Blog\RFM Analysis\Donations.csv"
OUT_FILE = r"C:\Ken\Blog\RFM Analysis\RFM Donors Snapshots.csv"

AS_OF_TODAY: date = datetime.now().date()

EXPECTED_COLUMNS = [
    "Donor ID",
    "Donor Segment",
    "Donor Full Name",
    "Donor Country",
    "Donor State",
    "Donor City",
    "Donor Zip",
    "Gift ID",
    "Gift Date",
    "Gift Amount",
    "Channel",
    "Campaign Name",
    "Fund/Designation",
    "Gift Type",
]

DONOR_COLUMNS = [
    "Donor ID",
    "Donor Segment",
    "Donor Full Name",
    "Donor Country",
    "Donor State",
    "Donor City",
    "Donor Zip",
]


# ============================================================
# VALIDATION
# ============================================================

def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input file does not match expected schema.\n"
            f"Missing columns: {missing}\n"
            f"Expected columns: {EXPECTED_COLUMNS}"
        )


# ============================================================
# SCORING UTILITIES
# ============================================================

def score_0_to_10(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Tableau Prep-style percentile rank scaled to 0–10:
      pct = (rank - 1) / (n - 1)
      score = pct * 10
    ties use rank(method='min')
    """
    s = pd.Series(series).astype(float)
    n = len(s)

    if n <= 1 or s.nunique(dropna=True) <= 1:
        return pd.Series([5.0] * n, index=s.index)

    ascending = True if higher_is_better else False
    r = s.rank(method="min", ascending=ascending)
    pct = (r - 1) / (n - 1)
    return pct * 10.0


def label_rfm_segments(r: float, f: float, m: float, rfm: float) -> str:
    # Your segment rules
    if r >= 9.5 and f >= 9.5 and m >= 9.5:
        return "Platinum"
    if r >= 9.0 and f >= 9.0 and m >= 9.0:
        return "Champions"
    if r >= 8.5 and f >= 8.0 and m >= 8.0:
        return "Potential Champions"
    if r >= 7.5 and f >= 7.5:
        return "Loyal"
    if m >= 9.0 and r >= 6.0:
        return "Big Spenders"
    if r >= 9.0 and f <= 2.5:
        return "New & Promising"
    if r <= 3.0 and (f >= 6.5 or m >= 6.5):
        return "At Risk"
    if r <= 2.5 and f <= 3.5 and m <= 3.5:
        return "Hibernating"
    if rfm >= 6.5:
        return "Above Average"
    if rfm >= 4.0:
        return "Average"
    return "Below Average"


# ============================================================
# DATE HELPERS
# ============================================================

def month_end_dates(min_gift_date: pd.Timestamp, max_gift_date: pd.Timestamp) -> pd.DatetimeIndex:
    min_month_start = min_gift_date.to_period("M").to_timestamp(how="start").normalize()
    max_month_start = max_gift_date.to_period("M").to_timestamp(how="start").normalize()

    month_starts = pd.date_range(min_month_start, max_month_start, freq="MS")
    month_ends = (month_starts + pd.offsets.MonthEnd(0)).normalize()
    return month_ends


# ============================================================
# RFM COMPUTATION
# ============================================================

def compute_rfm_fields(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    out = snapshot_df.copy()

    asof = pd.to_datetime(out["As of Date"]).dt.normalize()
    last = pd.to_datetime(out["Last Gift"]).dt.normalize()

    out["Days Since Last Gift"] = (asof - last).dt.days.astype(int)

    out["Total Gifts"] = pd.to_numeric(out["Total Gifts"], errors="raise").astype(int)
    out["Total Amount"] = pd.to_numeric(out["Total Amount"], errors="raise").astype(float)

    def _score_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["Recency Score"] = score_0_to_10(g["Days Since Last Gift"], higher_is_better=False)
        g["Frequency Score"] = score_0_to_10(g["Total Gifts"], higher_is_better=True)
        g["Monetary Score"] = score_0_to_10(g["Total Amount"], higher_is_better=True)
        g["RFM Score"] = (g["Recency Score"] + g["Frequency Score"] + g["Monetary Score"]) / 3.0
        g["RFM Segment"] = g.apply(
            lambda r: label_rfm_segments(
                r["Recency Score"],
                r["Frequency Score"],
                r["Monetary Score"],
                r["RFM Score"],
            ),
            axis=1,
        )
        return g

    out = out.groupby("As of Date", group_keys=False).apply(_score_group)
    out["Total Amount"] = out["Total Amount"].round(2)

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Load and validate input
    df = pd.read_csv(IN_FILE, encoding="utf-8")
    validate_schema(df)

    # Normalize / coerce types
    df["Donor ID"] = pd.to_numeric(df["Donor ID"], errors="raise").astype("int64")
    df["Gift Date"] = pd.to_datetime(df["Gift Date"], errors="raise").dt.normalize()
    df["Gift Amount"] = pd.to_numeric(df["Gift Amount"], errors="raise")

    # Donor dimension (stable donor attributes)
    donor_dim = (
        df.sort_values(["Donor ID", "Gift Date", "Gift ID"])
          .groupby("Donor ID", as_index=False)[DONOR_COLUMNS[1:]]
          .first()
    )

    # Aggregate gifts to daily donor totals
    daily = (
        df.groupby(["Donor ID", "Gift Date"], as_index=False)
          .agg(
              Day_Gifts=("Gift ID", "count"),
              Day_Amount=("Gift Amount", "sum"),
          )
    )

    # Compute cumulative totals per donor
    daily = daily.sort_values(["Donor ID", "Gift Date"])
    daily["Total Gifts"] = daily.groupby("Donor ID")["Day_Gifts"].cumsum()
    daily["Total Amount"] = daily.groupby("Donor ID")["Day_Amount"].cumsum()
    daily["Last Gift"] = daily.groupby("Donor ID")["Gift Date"].cummax()

    daily_cum = daily[["Donor ID", "Gift Date", "Last Gift", "Total Gifts", "Total Amount"]].copy()

    # Build snapshot dates (month-ends + today)
    min_gift = df["Gift Date"].min()
    max_gift = df["Gift Date"].max()

    snapshot_dates = list(month_end_dates(min_gift, max_gift))

    today_asof = pd.to_datetime(AS_OF_TODAY).normalize()
    if today_asof not in snapshot_dates:
        snapshot_dates.append(today_asof)

    snapshot_dates = pd.to_datetime(pd.Index(snapshot_dates)).sort_values()

    # Cartesian product: donors × snapshot dates
    snap = donor_dim[["Donor ID"]].merge(
        pd.DataFrame({"As of Date": snapshot_dates}),
        how="cross",
    )

    # Ensure correct dtypes for merge_asof
    snap["Donor ID"] = snap["Donor ID"].astype("int64")
    snap["As of Date"] = pd.to_datetime(snap["As of Date"]).dt.normalize()

    daily_cum["Donor ID"] = daily_cum["Donor ID"].astype("int64")
    daily_cum["Gift Date"] = pd.to_datetime(daily_cum["Gift Date"]).dt.normalize()

    # ------------------------------------------------------------
    # IMPORTANT: merge_asof requires the KEY columns globally sorted
    # Sort by the DATE key first, then the BY key.
    # Use a stable sort (mergesort) so ties behave consistently.
    # ------------------------------------------------------------
    left = snap.sort_values(["As of Date", "Donor ID"], kind="mergesort").reset_index(drop=True)
    right = daily_cum.sort_values(["Gift Date", "Donor ID"], kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="As of Date",
        right_on="Gift Date",
        by="Donor ID",
        direction="backward",
    )

    # Drop donors with no gifts as of that snapshot date
    merged = merged[merged["Total Gifts"].notna()].copy()

    # Flag current snapshot
    merged["Is Current"] = merged["As of Date"].dt.date == AS_OF_TODAY

    # Add donor attributes
    merged = merged.merge(donor_dim, on="Donor ID", how="left")

    # Compute RFM fields
    snapshot_rfm = compute_rfm_fields(merged)

    # Final column order
    out_cols = [
        "As of Date",
        "Is Current",
        "Donor ID",
        "Donor Segment",
        "Donor Full Name",
        "Donor Country",
        "Donor State",
        "Donor City",
        "Donor Zip",
        "Last Gift",
        "Days Since Last Gift",
        "Recency Score",
        "Total Gifts",
        "Frequency Score",
        "Total Amount",
        "Monetary Score",
        "RFM Score",
        "RFM Segment",
    ]

    out = snapshot_rfm[out_cols].copy()

    # Normalize dates for CSV consumers
    out["As of Date"] = pd.to_datetime(out["As of Date"]).dt.date
    out["Last Gift"] = pd.to_datetime(out["Last Gift"]).dt.date

    # Sort for readability
    out = out.sort_values(["As of Date", "Donor ID"]).reset_index(drop=True)

    # Write output
    out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"Snapshots file created → {OUT_FILE}")
    print(f"Rows written: {len(out):,}")


if __name__ == "__main__":
    main()
