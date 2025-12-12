import pandas as pd
import argparse


def extract_group(protein_str):
    """Extract group name (3rd token) from PDB_CHAIN_GROUP_RESNUM."""
    try:
        return protein_str.split("_")[2]
    except:
        return None


def strip_last_token(protein_str):
    """Remove the last underscore-separated token (e.g., trailing -158.89)."""
    parts = protein_str.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return protein_str


def sample_groups(input_csv, n=35, seed=1):
    # Correct reading with semicolon delimiter
    df = pd.read_csv(input_csv, sep=";", header=None, names=["protein", "values"])

    # Remove last token such as '-158.89'
    df["protein"] = df["protein"].apply(strip_last_token)

    # Extract group
    df["group"] = df["protein"].apply(extract_group)

    # Remove rows where group cannot be extracted
    df = df.dropna(subset=["group"])

    # Sample n per group
    sampled = (
        df.groupby("group", group_keys=False)
          .apply(lambda x: x.sample(n=min(n, len(x)), random_state=seed))
    )

    # Reorder columns
    sampled = sampled[["protein", "group", "values"]]

    # Save
    out_name = f"sample_detail_seed{seed}_size{n}.csv"
    sampled.to_csv(out_name, index=False)

    print(f"[✓] Saved: {out_name}")
    print(f"[✓] Rows sampled: {len(sampled)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--n", type=int, default=35, help="Sample size per group")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    sample_groups(args.csv, n=args.n, seed=args.seed)
