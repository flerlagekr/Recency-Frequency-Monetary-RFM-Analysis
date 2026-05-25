# RFM Analysis
Python code and Tableau Prep flows for performing RFM analysis on a list of donations.

Includes three versions of each. 

1) Traditional RFM - Creates scores of 1-5 and concatenated three-digit RFM score.
2) Modified RFM - Creates scores of 0-10 (not buckets) with a combined score that averages the 3. This allows for more nuance and better quantitative analysis.
3) Snapshot - Builds on the modified RFM, creating month-end snapshots. This allows for trend analysis on the RFM scores themselves.
