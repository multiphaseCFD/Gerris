#!/bin/bash
#SBATCH --account=def-bertrand
#SBATCH --gres=gpu:2              # Number of GPUs per node
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntask=2                 # Number of MPI process
#SBATCH --cpus-per-task=1         # CPU cores per MPI process
#SBATCH --mem=4G                  # memory per node
#SBATCH --time=0-03:00            # time (DD-HH:MM)
#SBATCH --mail-user=08cnbj@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load cuda/8.0.44
mpirun -np 2 ./main




