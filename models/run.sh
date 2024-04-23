#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=demo
#SBATCH --account=eecs592s001w24_class
#SBATCH --partition=spgpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=180g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/apalakod/eecs_592_project/Bert_training.out

# The application(s) to execute along with its input arguments and options:

/bin/hostname
nvidia-smi
python3 /home/apalakod/eecs_592_project/models/climate_denial_downstream.py --pretrained_model_path='/home/apalakod/eecs_592_project/zip_gg_files/fine-tuned_models/base' --model_name='base' --data_path='/home/apalakod/eecs_592_project/data/climate_sentiment_train.csv' --test_data_path='/home/apalakod/eecs_592_project/data/climate_sentiment_test.csv' --mode='test-base'

