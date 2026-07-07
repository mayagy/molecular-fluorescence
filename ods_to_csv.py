import pandas as pd

from pathlib import Path


def convert_all(data_dir: Path):
    ods_dir = data_dir / "ods"
    ods_files = sorted(ods_dir.glob("*.ods"))
    if not ods_files:
        print("No .ods files found in", data_dir)
        return

    for ods_path in ods_files:
        csv_path = data_dir / "csv" / ods_path.with_suffix(".csv").name
        df = pd.read_excel(ods_path, engine="odf")
        df.to_csv(csv_path, index=False)
        print(f"{ods_path.name} -> {csv_path.name}  ({len(df)} rows)")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    convert_all(data_dir)
