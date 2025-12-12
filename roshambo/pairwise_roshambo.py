import os
import argparse
import pandas as pd


def flatten_roshambo(input_dir, output_dir):

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "pairwise_flattened.csv")
    all_rows = []

    # -------------------------------------------------
    # PROCESS EACH *_roshambo.csv FILE
    # -------------------------------------------------
    for file in os.listdir(input_dir):
        if not file.endswith("_roshambo.csv"):
            continue

        csv_path = os.path.join(input_dir, file)
        print(f"[+] Processing {csv_path}")

        df = pd.read_csv(csv_path)

        # Extract required fields
        small_df = df[["Query", "Molecule", "ShapeTanimoto", "ComboTanimoto"]].copy()

        # Rename Molecule → TargetMolecule
        small_df.rename(columns={"Molecule": "TargetMolecule"}, inplace=True)

        all_rows.append(small_df)

    if not all_rows:
        print("[✗] No *_roshambo.csv found in directory.")
        return

    # -------------------------------------------------
    # COMBINE ALL ROWS
    # -------------------------------------------------
    final_df = pd.concat(all_rows, ignore_index=True)

    # -------------------------------------------------
    # REMOVE EXACT DUPLICATE ROWS
    # -------------------------------------------------
    final_df.drop_duplicates(subset=["Query", "TargetMolecule"], inplace=True)

    # -------------------------------------------------
    # REMOVE MIRRORED DUPLICATES
    # Example: A-B and B-A → keep only one
    # -------------------------------------------------
    final_df["pair_key"] = final_df.apply(
        lambda row: tuple(sorted([row["Query"], row["TargetMolecule"]])),
        axis=1
    )

    # Drop duplicates based on unordered pair
    final_df = final_df.drop_duplicates(subset=["pair_key"]).copy()

    final_df.drop(columns=["pair_key"], inplace=True)

    # -------------------------------------------------
    # SAVE FINAL FILE
    # -------------------------------------------------
    final_df.to_csv(output_file, index=False)

    print(f"[✓] Saved flattened + mirrored-removed file → {output_file}")
    print(f"[✓] Total rows after cleanup: {len(final_df)}")


# -----------------------------------------------------
# ARGUMENT PARSER
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten ROSHAMBO output and remove mirrored duplicates."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing *_roshambo.csv files"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where final pairwise CSV will be saved"
    )

    args = parser.parse_args()

    flatten_roshambo(args.input_dir, args.output_dir)

