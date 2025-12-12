from mpi4py import MPI
import argparse
import os
import shutil
import traceback
import pandas as pd
from pandas.errors import EmptyDataError
from roshambo.api import get_similarity_scores


# ------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="MPI Parallel ROSHAMBO Runner")
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--query_dir", required=True)
    ap.add_argument("--dataset_sdf", required=True)
    ap.add_argument("--rosh_out", required=True)
    return ap.parse_args()


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    args = parse_args()

    BASE_DIR    = args.base_dir
    OUTPUT_DIR  = args.output_dir
    QUERY_DIR   = args.query_dir
    DATASET_SDF = args.dataset_sdf
    ROSH_OUT    = args.rosh_out

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --------------------------------------------------------
    # Rank 0: ensure output folder exists
    # --------------------------------------------------------
    if rank == 0:
        os.makedirs(ROSH_OUT, exist_ok=True)
        if not os.path.exists(DATASET_SDF):
            raise FileNotFoundError(f"dataset.sdf not found at: {DATASET_SDF}")
    comm.Barrier()

    # --------------------------------------------------------
    # Rank 0 collects query files and broadcasts list
    # --------------------------------------------------------
    if rank == 0:
        query_files = sorted(
            f for f in os.listdir(QUERY_DIR) if f.endswith(".sdf")
        )
        print(f"[Rank 0] Found {len(query_files)} SDF queries.")
    else:
        query_files = None

    query_files = comm.bcast(query_files, root=0)

    # Round-robin distribution
    my_files = query_files[rank::size]
    print(f"[Rank {rank}] Processing {len(my_files)} queries.")

    # --------------------------------------------------------
    # Create rank-specific working directory
    # --------------------------------------------------------
    rank_workdir = os.path.join(OUTPUT_DIR, f"rank_{rank}")
    os.makedirs(rank_workdir, exist_ok=True)

    # Copy dataset.sdf once into rank folder
    local_dataset = os.path.join(rank_workdir, "dataset.sdf")
    if not os.path.exists(local_dataset):
        shutil.copy(DATASET_SDF, local_dataset)

    # --------------------------------------------------------
    # Process queries assigned to this rank
    # --------------------------------------------------------
    for sdf_name in my_files:

        query_name = sdf_name.replace(".sdf", "")
        src_query = os.path.join(QUERY_DIR, sdf_name)

        print(f"[Rank {rank}] → Running: {query_name}")

        # Copy query into worker's local directory
        local_query = os.path.join(rank_workdir, sdf_name)
        shutil.copy(src_query, local_query)

        # Local ROSHAMBO output files
        rosh_csv = os.path.join(rank_workdir, "roshambo.csv")
        hits_sdf = os.path.join(rank_workdir, "hits.sdf")
        mols_sdf = os.path.join(rank_workdir, "mols.sdf")

        # Final cleaned output in shared results folder
        cleaned_output = os.path.join(ROSH_OUT, f"{query_name}_roshambo.csv")

        # ----------------------------------------------------
        # Run ROSHAMBO for this query
        # ----------------------------------------------------
        try:
            get_similarity_scores(
                ref_file=sdf_name,
                dataset_files_pattern="dataset.sdf",
                ignore_hs=True,
                n_confs=0,
                use_carbon_radii=True,
                color=True,
                sort_by="ComboTanimoto",
                write_to_file=True,
                gpu_id=0,
                working_dir=rank_workdir,
            )
        except Exception:
            print(f"[Rank {rank}] ERROR during ROSHAMBO for {query_name}")
            traceback.print_exc()
            # Cleanup local files and continue
            for f in [local_query, rosh_csv, hits_sdf, mols_sdf]:
                if os.path.exists(f):
                    os.remove(f)
            continue

        # ----------------------------------------------------
        # Load resulting roshambo.csv safely
        # ----------------------------------------------------
        if not os.path.exists(rosh_csv) or os.path.getsize(rosh_csv) == 0:
            print(f"[Rank {rank}] WARNING: Empty or missing output for {query_name}")
            for f in [local_query, rosh_csv, hits_sdf, mols_sdf]:
                if os.path.exists(f):
                    os.remove(f)
            continue

        try:
            df = pd.read_csv(rosh_csv, sep="\t")
        except EmptyDataError:
            print(f"[Rank {rank}] WARNING: roshambo.csv empty for {query_name}")
            for f in [local_query, rosh_csv, hits_sdf, mols_sdf]:
                if os.path.exists(f):
                    os.remove(f)
            continue

        # Cleanup / formatting same as original
        if "Molecule" in df.columns:
            df = df.drop(columns=["Molecule"])
        if "OriginalName" in df.columns:
            df = df.rename(columns={"OriginalName": "Molecule"})
        df.insert(0, "Query", query_name)

        df.to_csv(cleaned_output, index=False)
        print(f"[Rank {rank}] ✔ Saved → {cleaned_output}")

        # ----------------------------------------------------
        # Cleanup local temp files from this query
        # ----------------------------------------------------
        for f in [local_query, rosh_csv, hits_sdf, mols_sdf]:
            if os.path.exists(f):
                os.remove(f)

    print(f"[Rank {rank}] All assigned queries completed.")

    # --------------------------------------------------------
    # GLOBAL CLEANUP: remove all rank_* directories on rank 0
    # --------------------------------------------------------
    comm.Barrier()  # ensure all ranks are finished

    if rank == 0:
        print("[Rank 0] Removing all rank_* working directories...")
        for i in range(size):
            rdir = os.path.join(OUTPUT_DIR, f"rank_{i}")
            if os.path.exists(rdir):
                shutil.rmtree(rdir, ignore_errors=True)
        print("[Rank 0] Cleanup complete.")

    print(f"[Rank {rank}] Done.")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
