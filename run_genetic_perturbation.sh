#!/bin/bash
#SBATCH --job-name=run_genetic_perturbation
#SBATCH --output=run_genetic_perturbation.out.txt
#SBATCH --error=run_genetic_perturbation.err.txt
#SBATCH --time=48:00:00
#SBATCH --partition=amnewman
#SBATCH --mem=64G

ml hdf5
export PYTHONPATH=""
conda activate state

date
/oak/stanford/groups/amnewman/mkang9/util/miniconda3/envs/state/bin/python run_genetic_perturbation.py \
     --input /oak/stanford/groups/amnewman/mkang9/developmental_hierarchy/cancer_analysis/TNBC_malignant_h5ad/Newman_2022_PINK100_srtobj.rds.h5ad \
     --gene SERBP1 
/oak/stanford/groups/amnewman/mkang9/util/miniconda3/envs/state/bin/python run_genetic_perturbation.py \
     --input /oak/stanford/groups/amnewman/mkang9/developmental_hierarchy/cancer_analysis/TNBC_malignant_h5ad/Newman_2022_PINK100_srtobj.rds.h5ad \
     --gene MRPL41 
/oak/stanford/groups/amnewman/mkang9/util/miniconda3/envs/state/bin/python run_genetic_perturbation.py \
     --input /oak/stanford/groups/amnewman/mkang9/developmental_hierarchy/cancer_analysis/TNBC_malignant_h5ad/Newman_2022_PINK100_srtobj.rds.h5ad \
     --gene TUBA1B  


/oak/stanford/groups/amnewman/mkang9/util/miniconda3/envs/state/bin/python run_genetic_perturbation.py \
     --input /oak/stanford/groups/amnewman/mkang9/developmental_hierarchy/cancer_analysis/TNBC_malignant_h5ad/Newman_2022_PINK100_srtobj.rds_downsampled_100.h5ad \
     --gene ANLN 


/oak/stanford/groups/amnewman/mkang9/util/miniconda3/envs/state/bin/python run_genetic_perturbation.py \
     --input /oak/stanford/groups/amnewman/mkang9/developmental_hierarchy/cancer_analysis/TNBC_malignant_h5ad/Newman_2022_PINK100_srtobj.rds_downsampled_100.h5ad \
     --gene ABHD11 