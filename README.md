This repository contains for the paper [Bootstrap State Representation using Style Transfer for Better Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/2207.07749)


The initial code structure is from [NeurIPS 2020 - Procgen competition](https://www.aicrowd.com/challenges/neurips-2020-procgen-competition).
## Get started 
Clone this repository: 

``git clone https://github.com/masud99r/thinker.git``

``mkdir thinker_results``

``cd thinker``

## Install
``conda create -n thinker python=3.7`` 

``conda activate thinker``

Install pytorch: 
``pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html``

Install Ray: 
``pip install ray[rllib]==1.8.0``

Install sklearn: 
``pip install sklearn``

Install procgen environment: 
``pip install gym==0.21.0``


Install procgen environment: 
``pip install procgen==0.9.2``



## Run

To run ``Thinker`` on Procgen Maze environment run:

``python trainer_thinker_stargan.py --algo thinker --env maze --seed 100 --epoch 0 --n_cluster 3 --n_epochs 500 --max_timestep 25000000 --result_dir ../thinker_results``

To run ``PPO`` on Procgen Maze environment run:

``python trainer_baselines.py --algo ppo --env maze --seed 100 --max_timestep 25000000 --result_dir ../thinker_results``

To run ``RAD`` with ``random crop`` data augmentation on Procgen Maze environment run:

``python trainer_baselines_rad.py --algo rad_crop --env maze --seed 100 --max_timestep 25000000 --result_dir ../thinker_results``

To run ``RAD`` with ``random cutout color`` data augmentation on Procgen Maze environment run:

``python trainer_baselines_rad.py --algo rad_cutout --env maze --seed 100 --max_timestep 25000000 --result_dir ../thinker_results``


## Debuging

- We conducted our experiments on A100 GPU with 40GB memory. Code can be run on other GPU with low memory. A common error is Cuda out of memory in low memory case. In that case, try reducing ``sgd_minibatch_size`` and ``train_batch_size``. A sample is given in
in ``experiments/thinker-stargan-procgen-small.yaml``.
- In case of limited memory, the training can stop after training generator. In that case, we suggest to rerun the code using the ``--epoch`` flag to 1, which will load the saved generator modules and start RL training.
- Issue with rate in resnet torch hub: https://github.com/pytorch/pytorch/issues/61755#issuecomment-885801511


## Contact
Please contact the author at rahman64@purdue.edu if you have any queries.

## Citation
If you use this code or data, please consider citing this paper:

```
@inproceedings{rahman2022bootstrap,
  title={Bootstrap State Representation using Style Transfer for Better Generalization in Deep Reinforcement Learning},
  author={Rahman, Md Masudur and Xue, Yexiang},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD 2022)},
  year={2022}
}
```