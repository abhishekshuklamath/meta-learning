#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=3
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00
#SBATCH --mem=0
#SBATCH --partition=cpu2021

####### Set environment variables ###############
export PATH=/home/abhishekkumar.shukla/software/miniconda3/bin:$PATH

####### Run your script #########################
python processorscript.py