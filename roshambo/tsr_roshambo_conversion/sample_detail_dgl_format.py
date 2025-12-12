import argparse

def extract_protein_id(entry):
    entry = entry.strip()

    if not entry:
        return entry

    # Remove everything after the first ";"
    left = entry.split(";", 1)[0]  # e.g., 4UFA_A_1302_ASP_ASP

    parts = left.split("_")

    # Expected: PDB, CHAIN, SEQNUM, RESIDUE, GROUP
    if len(parts) < 5:
        return left  # if malformed, keep as is

    pdb, chain, seqnum, residue, group = parts

    # Output: PDB_CHAIN_SEQNUM_RESIDUE
    return f"{pdb}_{chain}_{seqnum}_{residue}"


def process_csv(input_csv, output_csv):
    cleaned = []

    with open(input_csv, "r") as infile:
        for line in infile:
            if line.strip():
                cleaned.append(extract_protein_id(line))

    with open(output_csv, "w") as outfile:
        for item in cleaned:
            outfile.write(item + "\n")

    print(f"[✓] Saved cleaned protein IDs to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDB_CHAIN_SEQNUM_RESIDUE from formatted CSV file")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--out", required=True, help="Output CSV file")
    args = parser.parse_args()

    process_csv(args.csv, args.out)
