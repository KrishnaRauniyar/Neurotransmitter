import pandas as pd
import argparse

# Normalize CSV1 protein: already PDB_CHAIN_RES_SEQ
def normalize_csv1(protein):
    return protein.strip()

# Normalize CSV2 protein: convert PDB_CHAIN_SEQ_RES_GRP → PDB_CHAIN_RES_SEQ
def normalize_csv2(protein2):
    protein2 = protein2.strip()
    parts = protein2.split("_")
    if len(parts) < 5:
        return None
    pdb, chain = parts[0], parts[1]
    seqnum, residue = parts[2], parts[3]
    return f"{pdb}_{chain}_{residue}_{seqnum}"


def main(csv1, csv2, output_file):

    # ---------------- CSV1 ----------------
    df1 = pd.read_csv(csv1)
    df1["norm"] = df1["protein"].apply(normalize_csv1)

    print("\n===== CSV1 NORMALIZED =====")
    print(df1["norm"].head(10).tolist())

    # ---------------- CSV2 ----------------
    # IMPORTANT: CSV2 uses semicolon as delimiter
    df2 = pd.read_csv(csv2, header=None, sep=";", dtype=str, engine="python")
    df2.columns = ["protein2", "values2"]  # two columns: protein, values

    df2["protein2"] = df2["protein2"].str.strip()
    df2["norm"] = df2["protein2"].apply(normalize_csv2)

    print("\n===== CSV2 NORMALIZED =====")
    print(df2["norm"].head(10).tolist())

    # Remove rows where normalization failed
    df2 = df2[df2["norm"].notna()]

    # ---------------- MATCH ----------------
    merged = df1.merge(df2, on="norm", how="inner")

    print("\n===== MERGED MATCHES =====")
    print(merged[["protein", "protein2"]].head())

    # Output: protein2 ; values(from CSV1)
    merged["output"] = merged["protein2"] + ";" + merged["values"].str.rstrip(",")

    # Write exactly what we want: no quotes, just raw lines
    with open(output_file, "w") as f:
        for line in merged["output"]:
            f.write(str(line) + "\n")


    print("\nSaved:", output_file)
    print("Matched rows:", len(merged))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv1", required=True)
    parser.add_argument("--csv2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.csv1, args.csv2, args.output)
