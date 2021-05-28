#!/bin/bash
#SBATCH --job-name=script
#SBATCH --partition=mbzuai
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/projects/mbzuai/peterahn/workspace/substruct-embedding/resource/log/job_%j.log
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=8

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjY3ZDAyMWYtZWQ5MC00NGQwLTg4OWItN2U3YzU4YWE3YzJkIn0="

srun \
  --container-image=sungsahn0215/ogbg-transfer:latest \
  --no-container-mount-home \
  --container-mounts="/nfs/projects/mbzuai/peterahn/workspace/mol-selfsup:/mol-selfsup" \
  --container-workdir="/mol-selfsup/src" \
  bash ../script/${1}.sh