import pandas as pd
import argparse

def convert_format(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Reorder and append group again at the end
    def reorder_and_append(p):
        parts = p.split("_")
        if len(parts) != 4:
            return p
        pdb, chain, group, resnum = parts

        # Reorder → PDB_CHAIN_RESNUM_GROUP
        new_name = f"{pdb}_{chain}_{resnum}_{group}"

        # Append group again → PDB_CHAIN_RESNUM_GROUP_GROUP
        return f"{new_name}_{group}"

    df["protein"] = df["protein"].apply(reorder_and_append)

    # Save only the protein column
    df[["protein"]].to_csv(output_csv, index=False)
    print(f"[✓] Saved output to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat and append group to protein IDs")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    convert_format(args.input, args.output)



# python3 input_roshambo_rmsd.py --input sample_detail_seed1_size20.csv --output sample_detail_seed1_size20_rmsd.csv
