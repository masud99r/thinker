#!/bin/bash
hostname
python trainer_thinker.py --algo $1 --env $2 --seed $3 --load_trained $4 --resume $5 --n_cluster $6 --n_epochs $7 --max_timestep $8

# python trainer_thinker.py --algo thinker --env leaper --seed 100 --load_trained 0 --resume 0 --n_cluster 3 --n_epochs 5 --max_timestep 1000000
# python trainer_thinker.py --algo thinker --env leaper --seed 100 --load_trained 1 --resume 1 --n_cluster 3 --n_epochs 5 --max_timestep 2000000
# python trainer_thinker.py --algo thinker --env leaper --seed 100 --load_trained 1 --resume 1 --n_cluster 3 --n_epochs 5 --max_timestep 3000000

# sbatch --nodes=1 --mem=100G --constraint=G --gpus-per-node=1 --time=23:59:00 train_thinker.sh thinker jumper 100 0 25000000 20.0 20.0


# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 100 1 0 3 5 1000000
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 100 1 1 3 5 2000000
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 100 1 1 3 5 3000000

# sbatch --nodes=1 --mem=100G --constraint=G --gpus-per-node=1 --time=23:59:00 train_run.sh thinker leaper 100 0 0 3 5 25000000
# sbatch --nodes=1 --mem=100G --constraint=G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker climber 100 0 0 3 5 5000000
# sbatch --nodes=1 --mem=100G --constraint=G --gpus-per-node=1 --time=23:59:00 train_run.sh thinker climber 200 1 0 3 5 25000000

# sbatch --nodes=1 --mem=100G --constraint=G --gpus-per-node=1 --time=23:59:00 train_run.sh thinker leaper 300 1 0 3 5 25000000


# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker climber 100 0 0 3 5 1000000
# sbatch --nodes=1 --mem=100G --constraint=G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker climber 100 1 0 3 5 100000
# sbatch --nodes=1 --mem=100G --constraint=G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker climber 100 1 1 3 5 200000

# sbatch --nodes=1 --mem=200G --constraint=G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker fruitbot 100 1 1 3 5 5000000

# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker heist 100 0 0 3 5 3000000
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker dodgeball 100 0 0 3 5 3000000

# python trainer_thinker.py --algo thinker --env climber --seed 100 --load_trained 1 --resume 0 --n_cluster 3 --n_epochs 5 --max_timestep 100000

# sbatch --nodes=1  --gpus-per-node=1 --time=19:30:00 -E train_run.sh thinker leaper 100 1 1 3 5 2000000

# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh
# sbatch --nodes=1 --mem=150G --gpus-per-node=1 --time=19:30:00 train_run.sh
# sbatch --nodes=1 --mem=200G  --gpus-per-node=1 --constraint=B --time=19:30:00 train_run.sh
# sbatch --nodes=1 --constraint=C --time=19:30:00 train_run.sh

# tensorboard --host=0.0.0.0 --logdir=/scratch/gilbreth/rahman64/results/thinker_bootstrap/procgen
# tensorboard --host=0.0.0.0 --logdir=/scratch/gilbreth/rahman64/results/thinker_bootstrap/procgen/run_seeds


#run
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 100
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 200
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 300
# python trainer_thinker.py --algo thinker --env leaper --seed 200 --load_trained 1
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 100 1

# Thinker run on leaper
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 100 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 200 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker leaper 300 0

# PPO on leaper
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh ppo leaper 100 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh ppo leaper 200 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh ppo leaper 300 0

# Thinker run on dodgeball
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker dodgeball 100 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker dodgeball 200 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker dodgeball 300 0

# climber
# Thinker-PPO
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker climber 100 0

# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker climber 200 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh thinker climber 300 0

# PPO
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh ppo climber 100 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh ppo climber 200 0
# sbatch --nodes=1 --mem=100G --gpus-per-node=1 --time=19:30:00 train_run.sh ppo climber 300 0