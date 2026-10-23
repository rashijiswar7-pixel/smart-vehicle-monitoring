from pathlib import Path
import pandas as pd

# Locate the data folder
DATA_FOLDER = Path(__file__).parent / "data"

# Find all CSV files
csv_files = list(DATA_FOLDER.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV files found in the data folder.")

else:
    for file_path in csv_files:
        print("\n" + "=" * 70)
        print("FILE:", file_path.name)
        print("=" * 70)

        try:
            df = pd.read_csv(file_path, nrows=5)

            print("\nNumber of columns:", len(df.columns))
            print("\nColumn names:")
            print(df.columns.tolist())

            print("\nFirst 5 rows:")
            print(df.to_string(index=False))

            print("\nData types:")
            print(df.dtypes.to_string())

        except Exception as error:
            print("Could not read file:", error)