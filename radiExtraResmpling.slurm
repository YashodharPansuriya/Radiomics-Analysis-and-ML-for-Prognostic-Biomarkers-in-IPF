#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --output=RadiomicsExtractionresampling_Output.log
#SBATCH --error=RadiomicsCovidExtractionresampling_error.log
#SBATCH --time=200:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source /home/pansurya/.venv/bin/activate

python RadiomicExtactionResampling.py
