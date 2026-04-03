#!/bin/bash
#SBATCH --account=kernlab
#SBATCH --partition=kern
#SBATCH --job-name=Ecolocator
#SBATCH --time=7-0
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --error=job%j.err
#SBATCH --out=job%j.out

#geno_file=$1
#test_file=$2
#out_subdir=$3
#python ../ecolocator.py --matrix ${geno_file} --sample_data ${test_file} --out ${out_subdir} --env_weight 0.5 --loc_weight 0.5 --save_metrics --num_covs 1

#Nate's way
jobfile=$1
#while read geno_file test_file out_subdir model_file; do #for trained model run
while read geno_file test_file out_subdir; do
    #echo "Running ecolocator with $geno_file and $test_file and $out_subdir" # for testing
    python ../ecolocatorcopy1.py --matrix ${geno_file} --sample_data ${test_file} --env_weight 0.5 --loc_weight 0.5 --num_covs 1 --keep_model --out ${out_subdir} # for gametime ADD --load_train_model ${model_file} for running with trained model, otherwise leave --keep_model for saving model
done < $jobfile
