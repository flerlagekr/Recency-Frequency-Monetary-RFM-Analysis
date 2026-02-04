# Reads a raw donor gift file in the schema defined below,
# computes donor-level RFM fields, and writes an enriched CSV.

# Adds:
# - Last Gift
# - Days Since Last Gift
# - Recency Score - Based on Days Since Last Gift (lower is better)
# - Total Gifts
# - Frequency Score  based on Total Gifts
# - Total Amount
# - Monetary Score based on Total Amount
# - RFM (3-digit string)
# - RFM Segment (label)

from datetime import date, datetime
import pandas as pd

in_file =  r'C:\Ken\Blog\RFM Analysis\Donations.csv'
out_file = r'C:\Ken\Blog\RFM Analysis\RFM 1-5.csv'

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
            f"Expected columns (at minimum): {EXPECTED_COLUMNS}"
        )


def QuintileScores(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    # Convert numeric series into 1–5 scores using quintiles.
    # If higher_is_better=False, lower values receive higher scores.
    # Robust to ties/duplicates by ranking first.
    s = series.copy()

    if s.nunique(dropna=True) < 5:
        ranks = s.rank(method="average", na_option="keep")
        scores = pd.cut(ranks, bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True).astype("Int64")
    else:
        ranks = s.rank(method="average")
        scores = pd.qcut(ranks, q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    if not higher_is_better:
        scores = 6 - scores.astype(int)

    return scores.astype(int)


def LabelRFMSegments(r: int, f: int, m: int) -> str:
    """
    RFM Segment mapping with strict Champions definition.

    r, f, m are integers 1–5.

    Champions            = 5-5-5 only
    Potential Champions  = strong across all three, but not perfect
    Loyal                = frequent and recent
    Big Spenders         = very high monetary value
    New                  = very recent first-time donors
    Promising            = recent but low frequency
    At Risk              = slipping donors who were stronger before
    About to Sleep       = medium recency, low frequency
    Hibernating          = old + low engagement
    Other                = everything else
    """

    # Best of the best
    if r == 5 and f == 5 and m == 5:
        return "Champions"

    # Nearly elite
    if r >= 4 and f >= 4 and m >= 4:
        return "Potential Champions"

    # Consistent supporters
    if r >= 4 and f >= 4:
        return "Loyal"

    # High monetary but not necessarily frequent
    if m == 5 and r >= 3:
        return "Big Spenders"

    # Brand new donors
    if r == 5 and f == 1:
        return "New"

    # Recent but not yet frequent
    if r >= 4 and f <= 2:
        return "Promising"

    # Previously strong but not recent
    if r == 2 and (f >= 3 or m >= 3):
        return "At Risk"

    # Showing signs of disengagement
    if r == 3 and f <= 2:
        return "About to Sleep"

    # Long gone / very cold
    if r == 1 and f <= 2:
        return "Hibernating"

    return "Other"


def EnrichRFM() -> pd.DataFrame:
    df = pd.read_csv(in_file, encoding="utf-8")

    ValidateSchema(df)

    df["Gift Date"] = pd.to_datetime(df["Gift Date"], errors="raise")
    df["Gift Amount"] = pd.to_numeric(df["Gift Amount"], errors="raise")

    end_dt = pd.to_datetime(datetime.now().date())

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

    # Scores
    agg["Recency Score"]    = QuintileScores(agg["Days Since Last Gift"], higher_is_better=False)
    agg["Frequency Score"]  = QuintileScores(agg["Total Gifts"], higher_is_better=True)
    agg["Monetary Score"]   = QuintileScores(agg["Total Amount"], higher_is_better=True)

    agg["RFM Score"] = agg["Recency Score"].astype(str) + agg["Frequency Score"].astype(str) + agg["Monetary Score"].astype(str)

    agg["RFM Segment"] = agg.apply(
        lambda r: LabelRFMSegments(int(r["Recency Score"]), int(r["Frequency Score"]), int(r["Monetary Score"])),
        axis=1
    )

    out = df.merge(agg, on="Donor ID", how="left")

    # Nice column order: raw first, then derived
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
    return out


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    out = EnrichRFM()
    print(f"Wrote {len(out):,} rows to {out_file}")


if __name__ == "__main__":
    main()