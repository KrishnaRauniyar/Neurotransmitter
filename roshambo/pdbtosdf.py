import os
import argparse
import requests
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm


# ------------------------------------------------------------
# CLEANUP FUNCTION TO REMOVE EMPTY / BROKEN MOL BLOCKS
# ------------------------------------------------------------
def clean_sdf(sdf_path):
    temp_out = sdf_path + ".cleaned"

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    writer = Chem.SDWriter(temp_out)

    kept = 0
    removed = 0

    for mol in supplier:
        if mol is None or mol.GetNumAtoms() == 0:
            removed += 1
            continue
        if not mol.GetConformer().Is3D():
            removed += 1
            continue
        writer.write(mol)
        kept += 1

    writer.close()
    os.replace(temp_out, sdf_path)

    print(f"[✓] Cleaned SDF → {kept} valid molecules, {removed} removed")


# ------------------------------------------------------------
# SELECT SINGLE RESIDUE FROM PDB
# ------------------------------------------------------------
class ResidueSelect(Select):
    def __init__(self, chain_id, res_name, res_id):
        self.chain_id = chain_id
        self.res_name = res_name
        self.res_id = res_id

    def accept_residue(self, residue):
        return (
            residue.get_resname() == self.res_name and
            residue.id[1] == self.res_id and
            residue.get_parent().id == self.chain_id
        )


# ------------------------------------------------------------
# PARALLEL TSR PIPELINE
# ------------------------------------------------------------
class TSRPipelineParallel:
    def __init__(self, input_csv=None, single_protein=None, output_dir="output",
                 tool="rdkit", add_h=True, n_jobs=-1):

        self.input_csv = input_csv
        self.single_protein = single_protein
        self.output_dir = output_dir
        self.tool = tool.lower()
        self.add_h = add_h
        self.n_jobs = n_jobs

        self.pdb_dir = os.path.join(output_dir, "pdbs")
        self.sdf_dir = os.path.join(output_dir, "sdfs")
        self.dataset_file = os.path.join(output_dir, "dataset.sdf")

        os.makedirs(self.pdb_dir, exist_ok=True)
        os.makedirs(self.sdf_dir, exist_ok=True)

        if input_csv and os.path.exists(input_csv):
            self.df = pd.read_csv(input_csv)
        else:
            self.df = None

    # --------------------------------------------------------
    # DOWNLOAD PDB
    # --------------------------------------------------------
    def download_pdb(self, pdbid):
        url = f"https://files.rcsb.org/download/{pdbid}.pdb"
        save_path = os.path.join(self.pdb_dir, f"{pdbid}.pdb")

        if os.path.exists(save_path):
            return save_path

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return save_path
            else:
                print(f"[✗] Failed {pdbid}: {r.status_code}")
        except Exception as e:
            print(f"[✗] Error downloading {pdbid}: {e}")

        return None

    # --------------------------------------------------------
    # EXTRACT RESIDUE
    # --------------------------------------------------------
    def extract_residue(self, protein_str):
        try:
            pdbid, chain, res_name, num = protein_str.split("_")
            num = int(num)
        except:
            print(f"[✗] Bad entry: {protein_str}")
            return None

        pdb_path = os.path.join(self.pdb_dir, f"{pdbid}.pdb")
        if not os.path.exists(pdb_path):
            pdb_path = self.download_pdb(pdbid)
            if not pdb_path:
                return None

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdbid, pdb_path)

        out_name = f"{pdbid}_{chain}_{res_name}_{num}.pdb"
        out_path = os.path.join(self.pdb_dir, out_name)

        io = PDBIO()
        io.set_structure(structure)
        io.save(out_path, ResidueSelect(chain, res_name, num))

        return out_path

    # --------------------------------------------------------
    # RDKit PDB → SDF
    # --------------------------------------------------------
    def pdb_to_sdf_rdkit(self, pdb_path):
        sdf_name = os.path.basename(pdb_path).replace(".pdb", ".sdf")
        sdf_path = os.path.join(self.sdf_dir, sdf_name)

        mol = Chem.MolFromPDBFile(pdb_path, removeHs=not self.add_h)
        if mol is None:
            print(f"[✗] RDKit failed: {pdb_path}")
            return None
        ################# This does the embedding when H is added, which may mess up original coords #################
        if self.add_h:
            mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol)
        except:
            pass  # keep raw coords if embedding fails

        ############### This keeps original coords #################

        # Add hydrogens while preserving coordinates
        # if self.add_h:
        #     mol = Chem.AddHs(mol, addCoords=True)

        # Skip tiny/broken fragments
        if mol.GetNumAtoms() < 3:
            print(f"[✗] Skipping invalid (too few atoms): {pdb_path}")
            return None

        mol.SetProp("_Name", sdf_name.replace(".sdf", ""))

        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        return sdf_path

    # --------------------------------------------------------
    # OpenBabel PDB → SDF
    # --------------------------------------------------------
    def pdb_to_sdf_obabel(self, pdb_path):
        sdf_name = os.path.basename(pdb_path).replace(".pdb", ".sdf")
        sdf_path = os.path.join(self.sdf_dir, sdf_name)
        basename = sdf_name.replace(".sdf", "")
        ############# Preserves original geometry #############
        # cmd = ["obabel", "-ipdb", pdb_path, "-osdf", "-O", sdf_path]
        ############# Does not Preserves original geometry #############
        cmd = ["obabel", "-ipdb", pdb_path, "-osdf", "-O", sdf_path, "--gen3D"]
        if self.add_h:
            cmd.insert(1, "-h")
        else:
            cmd.append("-d")   # remove hydrogens

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            print(f"[✗] Open Babel failed for {pdb_path}")
            return None

    # -------- FIX TITLE LINE (IMPORTANT) --------
        with open(sdf_path, "r") as f:
            lines = f.readlines()

        # Replace FIRST line with the desired title
        lines[0] = basename + "\n"

        # Write back
        with open(sdf_path, "w") as f:
            f.writelines(lines)
        return sdf_path


    # --------------------------------------------------------
    # MERGE ALL SDFS
    # --------------------------------------------------------
    def merge_sdfs(self):
        writer = Chem.SDWriter(self.dataset_file)

        for sdf in os.listdir(self.sdf_dir):
            if not sdf.endswith(".sdf"):
                continue

            path = os.path.join(self.sdf_dir, sdf)
            mols = Chem.SDMolSupplier(path, removeHs=False)

            for mol in mols:
                if mol is None:
                    continue
                if mol.GetNumAtoms() == 0:
                    continue
                if not mol.GetConformer().Is3D():
                    continue
                writer.write(mol)

        writer.close()
        clean_sdf(self.dataset_file)
        print(f"[✓] Merged all → {self.dataset_file}")

    # --------------------------------------------------------
    # MAIN PARALLEL PIPELINE
    # --------------------------------------------------------
    def run(self):

        # SINGLE MODE
        if self.single_protein:
            try:
                pdbid, chain, res, num = self.single_protein.split("_")
                pdb_path = self.extract_residue(self.single_protein)

                if pdb_path:
                    if self.tool == "rdkit":
                        self.pdb_to_sdf_rdkit(pdb_path)
                    else:
                        self.pdb_to_sdf_obabel(pdb_path)
            except:
                print("Use format: PDB_CHAIN_RESNAME_RESNUM")
            return

        if self.df is None:
            print("No CSV provided")
            return

        # ----------------------------------------------------
        # 1. Download all PDBs
        # ----------------------------------------------------
        pdb_ids = sorted(set(v.split("_")[0] for v in self.df["protein"]))

        print(f"\n[1/4] Downloading {len(pdb_ids)} PDB files in parallel...")
        Parallel(n_jobs=self.n_jobs)(
            delayed(self.download_pdb)(pid) for pid in tqdm(pdb_ids , ascii=True)
        )

        # ----------------------------------------------------
        # 2. Extract residues
        # ----------------------------------------------------
        print(f"\n[2/4] Extracting residues in parallel...")
        pdb_paths = Parallel(n_jobs=self.n_jobs)(
            delayed(self.extract_residue)(protein_str)
            for protein_str in tqdm(self.df["protein"], ascii=True)
        )
        pdb_paths = [p for p in pdb_paths if p]

        # ----------------------------------------------------
        # 3. Convert to SDF
        # ----------------------------------------------------
        print(f"\n[3/4] Converting {len(pdb_paths)} PDBs to SDFs in parallel...")

        if self.tool == "rdkit":
            sdf_paths = Parallel(n_jobs=self.n_jobs)(
                delayed(self.pdb_to_sdf_rdkit)(p) for p in tqdm(pdb_paths, ascii=True)
            )
        else:
            sdf_paths = Parallel(n_jobs=self.n_jobs)(
                delayed(self.pdb_to_sdf_obabel)(p) for p in tqdm(pdb_paths, ascii=True)
            )

        sdf_paths = [s for s in sdf_paths if s]

        # ----------------------------------------------------
        # 4. Merge SDFs
        # ----------------------------------------------------
        print(f"\n[4/4] Merging all SDF files...")
        self.merge_sdfs()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_path")
    parser.add_argument("-s", "--single")
    parser.add_argument("-t", "--tool", default="rdkit", choices=["rdkit", "obabel"])
    parser.add_argument("--no_h", action="store_true")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-j", "--jobs", default=-1, type=int)
    args = parser.parse_args()

    TSRPipelineParallel(
        input_csv=args.input_path,
        single_protein=args.single,
        output_dir=args.output,
        tool=args.tool,
        add_h=not args.no_h,
        n_jobs=args.jobs
    ).run()
