#!/bin/bash
hostname
python trainer_thinker_stargan.py --algo $1 --env $2 --seed $3 --epoch $4 --n_cluster $5 --n_epochs $6 --max_timestep $7  --result_dir $8 # epoch = 0, then train from stratch, epoch = 1, then load trained model.


# python trainer_thinker_stargan.py --algo thinker --env maze --seed 100 --epoch 0 --n_cluster 3 --n_epochs 500 --max_timestep 25000000 --result_dir ../thinker_results

# sbatch --nodes=1 --mem=100G --constraint=G --gpus-per-node=1 --time=23:59:00 train_thinker_stargan.sh thinker maze 100 0 3 500 25000000 ../thinker_results
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=23:59:00 train_thinker_stargan.sh thinker maze 100 0 3 500 25000000 /home/rahman64/research/code-release/thinker_results