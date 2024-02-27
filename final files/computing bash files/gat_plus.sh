#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint='volta|ampere'



module load anaconda
source activate highway_gpu
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun python SAC_GNN_gpubuffer_GAT+.py