#!/usr/bin/env python
from mpi4py import rc
rc.thread_level = 'single'

import warnings
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message="urllib3 v2 only supports OpenSSL 1.1.1+"
)

import os
import argparse
import requests
from urllib.parse import urljoin
from mpi4py import MPI
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
import open3d as o3d
import datetime

def download_pdb(pdb_id, download_dir):
    pdb_id = pdb_id.lower()
    os.makedirs(download_dir, exist_ok=True)
    out = os.path.join(download_dir, f"{pdb_id}.pdb")
    if os.path.exists(out):
        return out
    url = urljoin("https://files.rcsb.org/download/", f"{pdb_id}.pdb")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        with open(out, 'wb') as f:
            f.write(resp.content)
        return out
    except Exception:
        return None

def filter_csv_by_residue(csv_path, residue_type=None):
    df = pd.read_csv(csv_path)
    df['drug_name'] = df['Protein'].str.split('_').str[-1]
    df = df.sort_values('drug_name')
    if residue_type:
        df = df[df['drug_name'] == residue_type]
        if df.empty:
            return None
    cols = df['Protein'].str.split('_', expand=True)
    df['protein'], df['chain'], df['drug_id'], df['drug_name'] = cols[0], cols[1], cols[2], cols[3]
    return df

def process_one(pdb_id, chain_id, drug_name, drug_id, pdb_dir, exclude_h):
    pdb_path = download_pdb(pdb_id, pdb_dir)
    if not pdb_path:
        return None

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        model     = next(structure.get_models())
    except (
        PDBConstructionException,  # missing or malformed atoms
        IndexError,                # bad line length
        ValueError,                # empty file or bad data
        StopIteration,             # no models at all
        KeyError,                  # missing dictionary keys
        RuntimeError,              # generator errors
        TypeError,                 # NoneType comparisons, etc.
        UnboundLocalError          # BioPython header‐parse bug
    ) as e:
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"[Rank {rank}] Skipping {pdb_id}: parse error: {e}")
        return None

    for ch in model:
        if ch.id == chain_id:
            for res in ch:
                if str(res.id[1]) == drug_id and res.get_resname() == drug_name:
                    pts = []
                    for atom in res:
                        if exclude_h and (atom.get_name().startswith('H') or atom.element=='H'):
                            continue
                        v = atom.get_vector()
                        pts.append((v[0], v[1], v[2]))
                    return pts or None
    return None

def compute_rmsd_pair(drug1, drug2, coords_dict):
    A = np.asarray(coords_dict[drug1])
    B = np.asarray(coords_dict[drug2])
    pcdA = o3d.geometry.PointCloud()
    pcdB = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(A)
    pcdB.points = o3d.utility.Vector3dVector(B)
    pcdA.translate(-pcdA.get_center())
    pcdB.translate(-pcdB.get_center())
    threshold = 1.0
    reg = o3d.pipelines.registration.registration_icp(
        pcdB, pcdA, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    B2 = np.asarray(pcdB.transform(reg.transformation).points)
    A2 = np.asarray(pcdA.points)
    d2 = np.min(np.linalg.norm(A2[None,:,:] - B2[:,None,:], axis=2), axis=1)
    return np.sqrt(np.mean(d2 * d2))

def main():
    parser = argparse.ArgumentParser(description="MPI-accelerated ICP RMSD")
    parser.add_argument("--csv", required=True, help="input CSV file path")
    parser.add_argument("--residue", help="filter by residue name")
    parser.add_argument("--pdb_dir", default="pdb_downloads", help="where to cache PDBs")
    parser.add_argument("--exclude-hydrogens", action="store_true",
                        help="exclude hydrogen atoms")
    parser.add_argument("--outdir", default="rank_outputs",
                        help="directory for per-rank and merged outputs")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # create output directory once
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    comm.Barrier()

    # record job start
    job_start = datetime.datetime.now()
    if rank == 0:
        print(f"JOB START: {job_start:%Y-%m-%d %H:%M:%S}")

    # Stage 1: distribute PDB parsing
    if rank == 0:
        df = filter_csv_by_residue(args.csv, args.residue)
        entries = [] if df is None else list(zip(
            df['protein'], df['chain'], df['drug_name'], df['drug_id']
        ))
    else:
        entries = None
    entries = comm.bcast(entries, root=0)

    local_coords = {}
    for i in range(rank, len(entries), size):
        pdb_id, ch, name, did = entries[i]
        key = f"{pdb_id}_{ch}_{did}_{name}"
        pts = process_one(pdb_id, ch, name, did, args.pdb_dir, args.exclude_hydrogens)
        if pts:
            local_coords[key] = pts

    # gather coords
    all_coords = comm.gather(local_coords, root=0)
    if rank == 0:
        coords = {}
        for d in all_coords:
            coords.update(d)
        keys = sorted(coords.keys())
    else:
        coords = None
        keys = None
    coords = comm.bcast(coords, root=0)
    keys  = comm.bcast(keys, root=0)

    m = len(keys)
    total_pairs = m * (m - 1) // 2

    # Stage 2: split pair indices
    start = (total_pairs * rank) // size
    end   = (total_pairs * (rank + 1)) // size

    cum = np.empty(m, dtype=np.int64)
    cum[0] = 0
    for i in range(1, m):
        cum[i] = cum[i-1] + (m - i)

    out_lines = []
    for i in range(m - 1):
        row_start = cum[i]
        row_end   = row_start + (m - i - 1)
        lo = max(row_start, start)
        hi = min(row_end,   end)
        if lo >= hi:
            continue
        for offset in range(lo - row_start, hi - row_start):
            j = i + 1 + offset
            rmsd = compute_rmsd_pair(keys[i], keys[j], coords)
            out_lines.append(f"{keys[i]}\t{keys[j]}\t{rmsd:.4f}")

    # write per-rank output
    rank_file = os.path.join(args.outdir, f"rmsd_rank{rank}.txt")
    with open(rank_file, 'w') as f:
        f.write("Struc1\tStruc2\tRMSD\n")
        for line in out_lines:
            f.write(line + "\n")

    comm.Barrier()

    # merge on rank 0
    if rank == 0:
        merged = os.path.join(args.outdir, "similarity_all.txt")
        header = "Struc1\tStruc2\tRMSD"
        with open(merged, 'w') as outf:
            outf.write(header + "\n")
            for r in range(size):
                fn = os.path.join(args.outdir, f"rmsd_rank{r}.txt")
                if not os.path.isfile(fn):
                    continue
                with open(fn) as rf:
                    for line in rf:
                        stripped = line.strip()
                        if not stripped or stripped == header:
                            continue
                        outf.write(line if line.endswith("\n") else line + "\n")
        print(f"Merged into {merged}")


        # ─── Generate clustermap input ──────────────────────────────
        print("\nGenerating clustermap_input.txt…")
        cm_file = os.path.join(args.outdir, 'clustermap_input.txt')

        # 1) Read in all pairwise RMSDs
        pairs = []
        with open(merged) as rf:
            next(rf)  # skip header
            for line in rf:
                s1, s2, val = line.strip().split('\t')
                try:
                    rmsd_val = float(val)
                except ValueError:
                    continue
                pairs.append((s1, s2, rmsd_val))

        # 2) Collect and sort the list of structures
        structs = sorted({s for (s,_,_) in pairs} | {t for (_,t,_) in pairs})
        n = len(structs)
        idx = {s: i for i, s in enumerate(structs)}

        # 3) Build an n×n symmetric matrix (0.0 on diagonal)
        rmsd_mat = [[0.0]*n for _ in range(n)]
        for s1, s2, v in pairs:
            i, j = idx[s1], idx[s2]
            rmsd_mat[i][j] = v
            rmsd_mat[j][i] = v

        # 4) Write out in “clustermap” form: each line is “STRUCT;v1,v2,…,vn”
        with open(cm_file, 'w') as outf:
            for i, s in enumerate(structs):
                row = ','.join(f"{rmsd_mat[i][j]:.4f}" for j in range(n))
                outf.write(f"{s};{row}\n")

        print(f"Clustermap input saved to {cm_file}")

    # record job end
    job_end = datetime.datetime.now()
    if rank == 0:
        duration = job_end - job_start
        print(f"JOB END:   {job_end:%Y-%m-%d %H:%M:%S}")
        print(f"DURATION: {duration}")

if __name__ == "__main__":
    main()
