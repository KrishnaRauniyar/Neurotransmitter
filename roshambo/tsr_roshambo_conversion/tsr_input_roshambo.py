import pandas as pd
import argparse

def process_file(input_csv, output_csv):
    data = []

    with open(input_csv, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split before semicolon
            left = line.split(";")[0]   # "4UFA_A_ASP_1302_3.94"
            parts = left.split("_")

            # Protein = everything except last (score)
            protein = "_".join(parts[:-1])

            # Group/residue = 3rd part (index 2)
            group = parts[2] if len(parts) >= 3 else ""

            data.append([protein, group])

    df = pd.DataFrame(data, columns=["protein", "group"])
    df.to_csv(output_csv, index=False)

    print(f"[✓] Saved output to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract protein and group columns")

    parser.add_argument("--input", required=True, help="Input CSV containing raw data")
    parser.add_argument("--output", required=True, help="Output CSV with protein, group")

    args = parser.parse_args()
    process_file(args.input, args.output)
