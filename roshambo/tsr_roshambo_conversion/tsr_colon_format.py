import pandas as pd
import argparse


def reorder_protein(name):
    """
    Convert PDB_CHAIN_RESNAME_RESNUM → PDB_CHAIN_RESNUM_RESNAME
    Example: 7U64_O_7V7_301 → 7U64_O_301_7V7
    """
    parts = name.split("_")
    if len(parts) != 4:
        return name
    pdb, chain, resname, resnum = parts
    return f"{pdb}_{chain}_{resnum}_{resname}"


def convert_csv(input_csv, output_file):
    df = pd.read_csv(input_csv)

    lines = []

    for _, row in df.iterrows():
        protein_raw = row["protein"]
        group = str(row["group"]).strip()
        values = str(row["values"]).strip('"')

        protein_new = reorder_protein(protein_raw)

        final = f"{protein_new}_{group};{values}"
        lines.append(final)

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✓] Saved formatted file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSR format converter")

    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output formatted TXT/CSV file")

    args = parser.parse_args()

    convert_csv(args.input, args.output)
