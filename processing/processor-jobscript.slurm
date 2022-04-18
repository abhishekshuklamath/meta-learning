#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00
#SBATCH --mem=10G
#SBATCH --partition=single

####### Set environment variables ###############
export PATH=/home/abhishekkumar.shukla/software/miniconda3/bin:$PATH

####### Run your script #########################
python processorscript.py