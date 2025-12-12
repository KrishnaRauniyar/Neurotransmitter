import pandas as pd
import argparse


def reorder_name(name):
    """Convert PDB_CHAIN_RESNAME_RESNUM → PDB_CHAIN_RESNUM_RESNAME."""
    parts = name.split("_")
    if len(parts) != 4:
        return name
    pdb, chain, resname, resnum = parts
    return f"{pdb}_{chain}_{resnum}_{resname}"


def convert_to_matrix(input_csv, output_csv, metric="ShapeTanimoto", no_header=False):
    print(f"[+] Loading file: {input_csv}")
    df = pd.read_csv(input_csv)

    # ----------------------------------------
    # 1. Normalize names
    # ----------------------------------------
    df["Query"] = df["Query"].apply(reorder_name)
    df["TargetMolecule"] = df["TargetMolecule"].apply(reorder_name)

    # ----------------------------------------
    # 2. Ensure metric exists
    # ----------------------------------------
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {df.columns}")

    # ----------------------------------------
    # 3. Keep only ID pairs inside square matrix
    # ----------------------------------------
    queries = set(df["Query"].unique())
    df = df[df["TargetMolecule"].isin(queries)]

    print(f"[INFO] Unique Query count: {len(queries)}")

    # ----------------------------------------
    # 4. Add symmetric missing pairs
    # ----------------------------------------
    print("[INFO] Adding missing symmetric pairs...")
    metric_map = dict()

    for q, t, v in df[["Query", "TargetMolecule", metric]].itertuples(index=False):
        metric_map[(q, t)] = v

    reverse_rows = []
    for q in queries:
        for t in queries:
            if (q, t) in metric_map and (t, q) not in metric_map:
                reverse_rows.append({"Query": t, "TargetMolecule": q, metric: metric_map[(q, t)]})

    if reverse_rows:
        df = pd.concat([df, pd.DataFrame(reverse_rows)], ignore_index=True)

    print(f"[INFO] Added {len(reverse_rows)} reverse rows.")

    # ----------------------------------------
    # 5. Build symmetric matrix
    # ----------------------------------------
    matrix = df.pivot_table(
        index="Query",
        columns="TargetMolecule",
        values=metric,
        fill_value=0.0,
    )

    matrix = matrix.reindex(index=sorted(queries), columns=sorted(queries), fill_value=0.0)

    # ----------------------------------------
    # 6. Convert similarity → distance
    # ----------------------------------------
    print("[INFO] Converting similarity → distance (1 - value)...")
    matrix = 1.0 - matrix
    matrix = matrix.clip(lower=0.0, upper=1.0)

    # Round only once
    matrix = matrix.round(3)

    # ----------------------------------------
    # 7. Save output
    # ----------------------------------------
    print("[+] Saving output...")
    if no_header:
        # Produce only no-header version
        noheader_path = output_csv.replace(".csv", "_noheader.csv")

        with open(noheader_path, "w") as f:
            for idx, row in matrix.iterrows():
                row_vals = ",".join(f"{v:.3f}" for v in row.values)
                f.write(f"{idx};{row_vals}\n")

        print(f"[✓] No-header CSV saved → {noheader_path}")

    else:
        # Save normal full CSV with headers
        matrix.to_csv(output_csv)
        print(f"[✓] Full matrix saved → {output_csv}")


# -------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True, help="Flattened Query–Target CSV")
    parser.add_argument("--output_csv", required=True, help="Output matrix CSV file")
    parser.add_argument("--metric", default="ShapeTanimoto", help="Metric to use")
    parser.add_argument("--no_header", action="store_true", help="Generate no-header file")

    args = parser.parse_args()

    convert_to_matrix(
        args.input_csv,
        args.output_csv,
        metric=args.metric,
        no_header=args.no_header
    )
