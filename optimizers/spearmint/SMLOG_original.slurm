#!/bin/bash
#SBATCH --job-name=log_reg_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="<s3305139@vuw.leidenuniv.nl>"
#SBATCH --mail-type="ALL"
#SBATCH --mem=5G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu-medium
#SBATCH --cpus-per-task=20



echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"

mkdir /scratchdata/datadir
touch /scratchdata/logfile

mongodb-linux-x86_64-rhel70-5.0.4/bin/mongod --fork --logpath /scratchdata/logfile --dbpath /scratchdata/datadir

python Spearmint/spearmint/main.py simple

echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"