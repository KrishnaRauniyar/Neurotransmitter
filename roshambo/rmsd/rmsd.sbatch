#!/bin/bash
#SBATCH -p workq
#SBATCH -N 84
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J rmsd_openmpi
#SBATCH -o logs/rmsd_%j.out
#SBATCH -e logs/rmsd_%j.err

set -e

# ensure logs/ exists
mkdir -p logs

module purge
module load conda/24.3.0

# ← Add this:
eval "$(conda shell.bash hook)"

ENV_DIR="$SLURM_SUBMIT_DIR/conda_openmpi"
if [ ! -d "$ENV_DIR" ]; then
  echo "Creating conda env at $ENV_DIR..."
  conda create -p "$ENV_DIR" \
    python=3.11 openmpi mpi4py biopython pandas psutil open3d numpy requests \
    scikit-learn \
  -c conda-forge -y
fi

# activate env 
conda activate "$ENV_DIR"

# prevent any threading beyond our 1 core per rank
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

outdir="$SLURM_SUBMIT_DIR/rank_outputs"
mkdir -p "$outdir"

echo "Starting $SLURM_NTASKS MPI ranks (1 core each) …"
srun --mpi=pmix --cpus-per-task=1 python $SLURM_SUBMIT_DIR/rmsd_mpi_gpu.py \
     --csv "/ddnB/work/rauniyar/Spring_2024/neurotransmitter/phase/rmsd/input_converter/output_1865.csv" \
     --pdb_dir $SLURM_SUBMIT_DIR/pdb_downloads \
     --exclude-hydrogens \
     --outdir "$outdir"

# merge on rank 0
if [ "$SLURM_PROCID" -eq 0 ]; then
  # concatenate from inside the output folder
  cat "$outdir"/rmsd_rank*.txt > "$outdir"/similarity_all.txt
  echo "Merged into $outdir/similarity_all.txt"
fi

conda deactivate
