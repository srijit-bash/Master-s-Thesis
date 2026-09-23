from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


# ========= CONFIG =========
INPUT_FOLDER = "."          # folder containing your CSV files
OUTPUT_MASTER = "slr_master_all.csv"  # merged file before dedup
OUTPUT_DEDUP = "slr_master_dedup.csv" # final deduplicated file
# =========================


def normalize_text(value: object) -> str:
    """Normalize text for safer duplicate comparison."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_doi(value: object) -> str:
    """Normalize DOI by lowercasing and removing URL prefixes."""
    text = normalize_text(value)
    text = text.replace("https://doi.org/", "")
    text = text.replace("http://doi.org/", "")
    text = text.replace("doi:", "")
    return text.strip()


def guess_query_name(filename: str) -> str:
    """Infer query label from filename."""
    name = filename.lower()
    if "ics" in name or "digital" in name:
        return "ICS"
    if "cyber" in name:
        return "CYBER"
    if "alarm" in name:
        return "ALARM"
    if "arch" in name or "framework" in name:
        return "ARCHITECTURE"
    return "UNKNOWN"


def combine_query_labels(series: pd.Series) -> str:
    """Combine overlapping query labels into one string."""
    labels = sorted({str(x).strip() for x in series if pd.notna(x) and str(x).strip()})
    return "; ".join(labels)


def first_nonempty(series: pd.Series) -> Optional[str]:
    """Return first non-empty value from a grouped column."""
    for val in series:
        if pd.notna(val) and str(val).strip():
            return str(val)
    return None


def load_and_merge_csvs(folder: Path) -> pd.DataFrame:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder.resolve()}")

    frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["Source_File"] = file.name
        df["Search_Query"] = guess_query_name(file.name)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    return merged


def deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize key columns expected from Scopus export
    required_cols = [
        "Authors",
        "Author full names",
        "Author(s) ID",
        "Title",
        "Year",
        "Source title",
        "Cited by",
        "DOI",
        "Link",
        "Abstract",
        "Author Keywords",
        "Index Keywords",
        "Source_File",
        "Search_Query",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["doi_norm"] = df["DOI"].apply(normalize_doi)
    df["title_norm"] = df["Title"].apply(normalize_text)

    # First, separate rows with DOI and without DOI
    with_doi = df[df["doi_norm"] != ""].copy()
    without_doi = df[df["doi_norm"] == ""].copy()

    # Deduplicate rows with DOI using DOI
    grouped_doi = (
        with_doi.groupby("doi_norm", dropna=False)
        .agg({
            "Authors": first_nonempty,
            "Author full names": first_nonempty,
            "Author(s) ID": first_nonempty,
            "Title": first_nonempty,
            "Year": first_nonempty,
            "Source title": first_nonempty,
            "Cited by": first_nonempty,
            "DOI": first_nonempty,
            "Link": first_nonempty,
            "Abstract": first_nonempty,
            "Author Keywords": first_nonempty,
            "Index Keywords": first_nonempty,
            "Search_Query": combine_query_labels,
            "Source_File": combine_query_labels,
            "title_norm": first_nonempty,
        })
        .reset_index(drop=True)
    )

    # Deduplicate rows without DOI using normalized title
    grouped_title = (
        without_doi.groupby("title_norm", dropna=False)
        .agg({
            "Authors": first_nonempty,
            "Author full names": first_nonempty,
            "Author(s) ID": first_nonempty,
            "Title": first_nonempty,
            "Year": first_nonempty,
            "Source title": first_nonempty,
            "Cited by": first_nonempty,
            "DOI": first_nonempty,
            "Link": first_nonempty,
            "Abstract": first_nonempty,
            "Author Keywords": first_nonempty,
            "Index Keywords": first_nonempty,
            "Search_Query": combine_query_labels,
            "Source_File": combine_query_labels,
        })
        .reset_index(drop=False)
    )

    dedup = pd.concat([grouped_doi, grouped_title], ignore_index=True)

    # Add helpful SLR columns
    dedup["Overlap_Count"] = dedup["Search_Query"].apply(
        lambda x: len(str(x).split("; ")) if pd.notna(x) and str(x).strip() else 0
    )
    dedup["Title_Screen"] = ""
    dedup["Abstract_Screen"] = ""
    dedup["Reason_for_Exclusion"] = ""
    dedup["Final_Inclusion"] = ""

    # Neater column order
    ordered_cols = [
        "Title",
        "Year",
        "Authors",
        "Source title",
        "DOI",
        "Cited by",
        "Link",
        "Abstract",
        "Author Keywords",
        "Index Keywords",
        "Search_Query",
        "Source_File",
        "Overlap_Count",
        "Title_Screen",
        "Abstract_Screen",
        "Reason_for_Exclusion",
        "Final_Inclusion",
    ]

    for col in ordered_cols:
        if col not in dedup.columns:
            dedup[col] = pd.NA

    dedup = dedup[ordered_cols].sort_values(
        by=["Overlap_Count", "Cited by", "Year"],
        ascending=[False, False, False],
        na_position="last"
    )

    return dedup


def main() -> None:
    folder = Path(INPUT_FOLDER)
    merged = load_and_merge_csvs(folder)
    merged.to_csv(OUTPUT_MASTER, index=False)

    dedup = deduplicate_records(merged)
    dedup.to_csv(OUTPUT_DEDUP, index=False)

    print("Done.")
    print(f"Merged rows: {len(merged)}")
    print(f"Deduplicated rows: {len(dedup)}")
    print(f"Saved merged file to: {Path(OUTPUT_MASTER).resolve()}")
    print(f"Saved deduplicated file to: {Path(OUTPUT_DEDUP).resolve()}")


if __name__ == "__main__":
    main()