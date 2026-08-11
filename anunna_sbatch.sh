#!/bin/bash
#SBATCH--cpus-per-task=16
#SBATCH--partition=gpu
#SBATCH--gpus=1
#SBATCH--job-name=PatchMatchNN
#SBATCH--mem=100G
#SBATCH--time=1000

cd /home/WUR/tijun001/PatchMatchNN
source vPatchMatchNN/bin/activate

# PREPARATION
#srun python3 ./main_data_preparation.py config/setup_1_full.json

# TRAINING
files=("config/setup_1.json" "config/setup_1_4.json" "config/setup_1_4a.json")
  # or
#files=(config/setup_*.json)
skip_files=("config/setup_8.json")sa

for file in "${files[@]}"; do
    if [[ " ${skip_files[*]} " =~ " $file " ]]; then
        echo "Skipping $file"
        continue
    fi
    echo "Training the model from $file"
    srun python3 ./main_training.py "$file" PatchMatchNN_trial_runs
done

# SSL PRETRAINING
#srun python3 ./main_ssl_training.py config/setup_8.json
#srun python3 ./main_ssl_training.py config/setup_9.json

#file=config/setup_11_1.json
#srun python3 ./main_ssl_training.py $file PatchMatchNN3.0
#srun python3 ./main_testing.py $file --acc_mode=validate_acc
#srun python3 ./main_testing.py $file --acc_mode=validate_top_3_acc

#file=config/setup_11_2.json
#srun python3 ./main_ssl_training.py $file PatchMatchNN3.0
#srun python3 ./main_testing.py $file --acc_mode=validate_acc
#srun python3 ./main_testing.py $file --acc_mode=validate_top_3_acc

# TESTING
#for file in  "${files[@]}"; do
#     if [[ " ${skip_files[*]} " =~ " $file " ]]; then
#        echo "Skipping $file"
#        continue
#     fi
#    echo "Testing the model from $file"
#    srun python3 ./main_testing.py $file --acc_mode=validate_acc
#    srun python3 ./main_testing.py $file --acc_mode=validate_top_3_acc
#done

## Top-k testing
#for file in config/setup_*.json; do
#    echo "Calculating top-k for $file"
#    #srun python3 ./main_topk.py $file
#done

# Individual
#file=config/setup_12_1.json
#project=PatchMatchNN3.0
#
#echo "Training the model from $file"
#srun python3 ./main_training.py $file $project
#
#echo "Testing the model from $file"
#srun python3 ./main_testing.py $file --acc_mode=validate_acc
#srun python3 ./main_testing.py $file --acc_mode=validate_top_3_acc

#file=config/setup_12.json
#echo "Calculating top-k for $file"
#srun python3 ./main_topk.py $file