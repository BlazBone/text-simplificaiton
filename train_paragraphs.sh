#!/bin/sh
#SBATCH --job-name=train_paragraphs
#SBATCH --output=together_.out
#SBATCH --error=together_.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-gpu=32G


source ~/miniconda3/etc/profile.d/conda.sh
conda activate new 


# script is made to run on 1 node with 2 gpus
srun --nodes 1 --exclusive --gpus=2 --ntasks=1 \
python3 ./code/basic.py \
--model-name cjvt/t5-sl-small \
--output-dir ./output \
--batch-size 16 \
--num-epochs 40 \
--learning-rate 0.0005 \
--data-dir ./data/paragraphs \
--token-len 512 &> ./t5-sl-small.log &

srun --nodes 1 --exclusive --gpus=2 --ntasks=1 \
python3 ./code/basic.py \
--model-name cjvt/t5-sl-large \
--output-dir ./output \
--batch-size 4 \
--num-epochs 40 \
--learning-rate 0.0005 \
--data-dir ./data/paragraphs \
--token-len 512 &> ./t5-sl-large.log &

srun --nodes 1 --exclusive --gpus=2 --ntasks=1 \
python3 ./code/basic.py \
--model-name google/mt5-base \
--output-dir ./output \
--batch-size 4 \
--num-epochs 40 \
--learning-rate 0.001 \
--data-dir ./data/paragraphs \
--token-len 512 &> ./mt5-base.log &

srun --nodes 1 --exclusive --gpus=2 --ntasks=1 \
python3 ./code/basic.py \
--model-name facebook/mbart-large-50 \
--output-dir ./output \
--batch-size 4 \
--num-epochs 40 \
--learning-rate 3e-5 \
--data-dir ./data/paragraphs  \
--token-len 512 &> ./mbart-large-50.log &

wait