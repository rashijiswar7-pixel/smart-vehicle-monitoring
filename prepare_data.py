from pathlib import Path
import pandas as pd

DATA_FOLDER = Path(__file__).parent / "data"
OUTPUT_FOLDER = DATA_FOLDER / "cleaned"

OUTPUT_FOLDER.mkdir(exist_ok=True)

csv_files = list(DATA_FOLDER.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV files found.")
else:
    for file_path in csv_files:
        print(f"\nProcessing: {file_path.name}")

        try:
            df = pd.read_csv(file_path, low_memory=False)

            # Remove leading/trailing spaces from column names
            df.columns = df.columns.str.strip()

            # Remove completely empty rows and columns
            df.dropna(how="all", inplace=True)
            df.dropna(axis=1, how="all", inplace=True)

            # Parse timestamps when available
            if "TIMESTAMP" in df.columns:
                df["TIMESTAMP"] = pd.to_datetime(
                    df["TIMESTAMP"], errors="coerce"
                )

            # Convert likely numeric columns carefully
            for column in df.columns:
                if column in ["TIMESTAMP", "TROUBLE_CODES", "DTC_NUMBER"]:
                    continue

                if df[column].dtype == "object":
                    values = df[column].astype(str).str.strip()

                    # Convert decimal commas to decimal points only
                    # when the value looks numeric.
                    numeric_values = pd.to_numeric(
                        values.str.replace(",", ".", regex=False)
                              .str.replace("%", "", regex=False),
                        errors="coerce"
                    )

                    # Replace the column only if most non-empty values
                    # can be interpreted as numbers.
                    non_empty = values.ne("") & values.ne("nan")
                    if non_empty.sum() > 0:
                        valid_ratio = numeric_values[non_empty].notna().mean()
                        if valid_ratio >= 0.8:
                            df[column] = numeric_values

            output_path = OUTPUT_FOLDER / f"{file_path.stem}_cleaned.csv"
            df.to_csv(output_path, index=False)

            print("Rows:", len(df))
            print("Columns:", len(df.columns))
            print("Saved:", output_path)

        except Exception as error:
            print("ERROR:", error)

print("\nData preparation finished.")