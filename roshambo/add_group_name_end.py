import pandas as pd

# ------------------------------------------------------------
# Normalize protein name
# sample file format:     PDB_CHAIN_RESNAME_RESNUM
# final_matrix format:    PDB_CHAIN_RESNUM_RESNAME
# ------------------------------------------------------------
def normalize_sample_name(name):
    """
    Convert from PDB_CHAIN_RESNAME_RESNUM → PDB_CHAIN_RESNUM_RESNAME
    Example:
        1BGV_A_GLU_501 → 1BGV_A_501_GLU
    """
    parts = name.split("_")
    if len(parts) != 4:
        return name
    pdb, chain, resname, resnum = parts
    return f"{pdb}_{chain}_{resnum}_{resname}"


# ------------------------------------------------------------
# Main function: append group names to final matrix
# ------------------------------------------------------------
def add_group_to_matrix(matrix_csv, sample_details_csv, output_csv):
    print("Loading sample details:", sample_details_csv)
    df_sample = pd.read_csv(sample_details_csv)

    # Ensure required columns exist
    if "protein" not in df_sample.columns or "group" not in df_sample.columns:
        raise ValueError("sample_details file must contain 'protein' and 'group' columns")

    # Build mapping: normalized_name → group
    mapping = {}
    for _, row in df_sample.iterrows():
        original = row["protein"]
        group = row["group"]

        # Normalize sample name so it matches final_matrix
        normalized = normalize_sample_name(original)

        mapping[normalized] = group

    print(f"[✓] Loaded {len(mapping)} protein → group mappings")

    # Process final matrix
    print("Processing matrix:", matrix_csv)
    with open(matrix_csv, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        if ";" not in line:
            new_lines.append(line)
            continue

        name, values = line.split(";", 1)

        # Attach group name if available
        if name in mapping:
            new_name = f"{name}_{mapping[name]}"
        else:
            new_name = name  # keep unchanged if no match

        new_lines.append(f"{new_name};{values}")

    # Write updated file
    print("Saving output:", output_csv)
    with open(output_csv, "w") as f:
        f.writelines(new_lines)

    print("[✓] Done! Output written to:", output_csv)


# ------------------------------------------------------------
# CLI execution
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Append group names to final_matrix CSV")
    parser.add_argument("--matrix", required=True, help="Path to final_matrix.csv")
    parser.add_argument("--detail", required=True, help="Path to sample_details CSV")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()

    add_group_to_matrix(
        matrix_csv=args.matrix,
        sample_details_csv=args.detail,
        output_csv=args.output
    )
