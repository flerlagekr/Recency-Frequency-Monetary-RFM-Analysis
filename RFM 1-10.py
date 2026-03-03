# Reads a raw donor gift file in the schema defined below,
# computes donor-level RFM fields, and writes an enriched CSV.
# Recency/Frequency/Monetary are scored from 0 to 10 (floats), based on rank.
# Combined RFM Score is the average of R, F, and M (0–10).

# Adds:
# - Last Gift
# - Days Since Last Gift
# - Recency Score - Based on Days Since Last Gift (lower is better)
# - Total Gifts
# - Frequency Score  based on Total Gifts
# - Total Amount
# - Monetary Score based on Total Amount
# - RFM
# - RFM Segment (label)


from datetime import datetime, date
import pandas as pd

# ============================================================
# CONFIGURATION (EDIT THESE)
# ============================================================

in_file =  r'C:\Ken\Blog\RFM Analysis\Donations.csv'
out_file = r'C:\Ken\Blog\RFM Analysis\RFM Modified.csv'

AS_OF_DATE = datetime.now().date()   # recency is calculated relative to this date

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


def ValidateSchema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input file does not match expected schema.\n"
            f"Missing columns: {missing}\n"
            f"Expected columns: {EXPECTED_COLUMNS}"
        )


def Score0to10(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    # Tableau Prep-style percentile rank scoring scaled to 0–10.
    #
    # Prep behavior:
    #   pct = (rank - 1) / (n - 1)
    #   ties: rank rounds down -> method='min'

    # higher_is_better=True:
    #   highest value -> 10, lowest -> 0

    # higher_is_better=False:
    #   lowest value -> 10, highest -> 0  (useful for Days Since Last Gift)
   
    s = pd.Series(series).astype(float)
    n = len(s)

    # If everything is the same (or only one row), there is no rank information.
    # Return a neutral midpoint or 0; midpoint tends to behave nicer for labeling.
    if n <= 1 or s.nunique(dropna=True) <= 1:
        return pd.Series([5.0] * n, index=s.index)

    # If higher_is_better, we want higher values to have higher pct -> sort ascending
    # If lower_is_better, we want lower values to have higher pct -> sort descending
    ascending = True if higher_is_better else False

    r = s.rank(method="min", ascending=ascending)  # Prep tie behavior
    pct = (r - 1) / (n - 1)                        # 0..1
    return pct * 10.0


def LabelRFMSegments(r: float, f: float, m: float, rfm: float) -> str:
    # Segment labels based on continuous 0–10 scores.
    # This is intentionally simple. Adjust thresholds to meet your needs.
    
    # The top of the heap.
    if r >= 9.5 and f >= 9.5 and m >= 9.5:
        return "Platinum"

    # Top tier
    if r >= 9.0 and f >= 9.0 and m >= 9.0:
        return "Champions"

    # Very strong overall
    if r >= 8.5 and f >= 8.0 and m >= 8.0:
        return "Potential Champions"

    # High frequency + recent (even if monetary isn't elite)
    if r >= 7.5 and f >= 7.5:
        return "Loyal"

    # High value donors (money) who are at least reasonably recent
    if m >= 9.0 and r >= 6.0:
        return "Big Spenders"

    # Very recent, low frequency
    if r >= 9.0 and f <= 2.5:
        return "New & Promising"

    # Slipping: not recent, but used to be meaningful
    if r <= 3.0 and (f >= 6.5 or m >= 6.5):
        return "At Risk"

    # Cold + low engagement
    if r <= 2.5 and f <= 3.5 and m <= 3.5:
        return "Hibernating"

    # Broad buckets for everything else
    if rfm >= 6.5:
        return "Above Average"
    if rfm >= 4.0:
        return "Average"
    else:
        return "Below Average"


def main():
    df = pd.read_csv(in_file, encoding="utf-8")
    ValidateSchema(df)

    df["Gift Date"] = pd.to_datetime(df["Gift Date"], errors="raise")
    df["Gift Amount"] = pd.to_numeric(df["Gift Amount"], errors="raise")

    end_dt = pd.to_datetime(AS_OF_DATE)

    agg = (
        df.groupby("Donor ID")
          .agg(
              **{
                  "Last Gift": ("Gift Date", "max"),
                  "Total Gifts": ("Gift ID", "count"),
                  "Total Amount": ("Gift Amount", "sum"),
              }
          )
          .reset_index()
    )

    agg["Days Since Last Gift"] = (end_dt - agg["Last Gift"]).dt.days.astype(int)

    # Continuous 0–10 scores
    agg["Recency Score"]    = Score0to10(agg["Days Since Last Gift"], higher_is_better=False)
    agg["Frequency Score"]  = Score0to10(agg["Total Gifts"], higher_is_better=True)
    agg["Monetary Score"]   = Score0to10(agg["Total Amount"], higher_is_better=True)

    # Combined numeric RFM (0–10)
    agg["RFM Score"] = (agg["Recency Score"] + agg["Frequency Score"] + agg["Monetary Score"]) / 3.0

    # Segment labels based on continuous scores
    agg["RFM Segment"] = agg.apply(
        lambda r: LabelRFMSegments(r["Recency Score"], r["Frequency Score"], r["Monetary Score"], r["RFM Score"],), axis=1)

    out = df.merge(agg, on="Donor ID", how="left")

    ordered_cols = EXPECTED_COLUMNS + [
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

    out = out[ordered_cols].sort_values(["Donor ID", "Gift Date", "Gift ID"]).reset_index(drop=True)
    out["Total Amount"] = out["Total Amount"].round(2)

    out.to_csv(out_file, index=False, encoding="utf-8")
    print(f"RFM file created → {out_file}")


if __name__ == "__main__":
    main()
