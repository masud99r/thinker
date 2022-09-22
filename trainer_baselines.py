#!/usr/bin/env python
from typing import Dict, List, Type, Union
from ray.rllib.utils.typing import TensorType
from ray.rllib.agents.trainer_template import build_trainer
# from .policy import RandomPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy import Policy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ppo_surrogate_loss
from ray.tune.logger import pretty_print

from typing import Optional, Type
from ray.rllib.utils.typing import TrainerConfigDict

from ray import tune
import os
from utils.loader import load_envs, load_models, load_algorithms, load_preprocessors
import yaml

#  clustering and cyclegan
import ray
import torch
from cluster import Cluster
from generator_ray import DenseCycleGAN
from os.path import join, exists
import os 
from os import mkdir
import random
import argparse

from data_augs_procgen import Rand_Crop, Cutout_Color



ray.init(num_cpus=20, num_gpus=1) # Only call this once.

# Fix numeric divergence due to bug in Cudnn # need to check

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)
# device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--algo", default='ppo', help="RL algorithm to use")
parser.add_argument("--env", default='leaper', help="RL env to use")
parser.add_argument("--seed", type=int, default=100, help="Seed for this run")
parser.add_argument("--load_trained", type=int, default=0, help="Whether load trained generator. 0 = False, 1 = True")
parser.add_argument("--resume", type=int, default=0, help="Whether resume RL training, considering the cluster and generators are trained. 0 = False, 1 = True")
parser.add_argument("--n_cluster", type=int, default=3, help="number of image channels")
parser.add_argument("--n_epochs", type=int, default=5, help="Total training epoch of generator")
parser.add_argument("--max_timestep", type=int, default=4000000, help="size of each image dimension")
parser.add_argument("--result_dir", default='./', help="directory to save results")


opt = parser.parse_args()
print(opt)

algo = opt.algo
n_epochs=opt.n_epochs
n_cluster = opt.n_cluster
max_timestep = opt.max_timestep


if opt.resume == 1:
    resume = True
else:
    resume = False

if opt.load_trained == 1:
    load_trained = True
else:
    load_trained = False

if resume: # override for resume, generator must be trained before this run
    load_trained = True 

if load_trained:
    load_epoch = n_epochs - 1 # last epoch
else:
    load_epoch = 0

f = 'experiments/thinker-stargan-procgen.yaml'

with open(f, 'r') as stream:
    try:
        experiments = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# experiments = yaml.safe_load(f)
experiments['env_config']['env_name'] = opt.env
experiments['evaluation_config']['env_config']['env_name'] = opt.env
experiments['seed'] = opt.seed

env_name = experiments['env_config']['env_name']
start_level = experiments['env_config']['start_level']
num_levels = experiments['env_config']['num_levels']
seed = experiments['seed']

print('experiments', experiments)

exp_name = algo+"_"+env_name+"_"+str(start_level)+"_"+str(num_levels)+"_seed_"+str(seed)+"_"+str(n_cluster)+"_maxtimestep_"+str(max_timestep)
print('exp_name', exp_name)

result_dir = opt.result_dir #"../thinker_results"

if not exists(result_dir):
    mkdir(result_dir)

# save in global trajectory_obs 
trajectory_obs = None
def convert_obs_rad(obs, aug='crop'):
    rand_crop = Rand_Crop(batch_size=obs.shape[0])
    # obs = obs.permute(0, 3, 1, 2)
    obs = obs.cpu().detach().numpy()
    # print('before obs', obs.shape)
    if aug == 'crop':
        obs = rand_crop.do_augmentation(obs)
    obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
    # obs = obs.permute(0, 2, 3, 1)
    # print('after crop obs', obs.shape)
    del rand_crop
    return obs

def agent_loss(policy, model, dist_class, train_batch):

    # change observation (current), train_batch
    if algo == 'rad_crop': # algo is a data augmentation
        obs = convert_obs_rad(train_batch[SampleBatch.CUR_OBS], aug='crop')
        train_batch[SampleBatch.CUR_OBS] = obs #.to(device)
        loss = ppo_surrogate_loss(policy, model, dist_class, train_batch) # average over original obs loss?
    # elif algo == 'mixreg':
    #     train_batch = convert_train_batch(algo, train_batch)
    #     loss = ppo_surrogate_loss(policy, model, dist_class, train_batch) # average over original obs loss?
    else: # ppo
        loss = ppo_surrogate_loss(policy, model, dist_class, train_batch) # default ppo
    return loss
AgentPolicy = PPOTorchPolicy.with_updates(
    name="AgentPolicy",
    loss_fn=agent_loss
    )
def get_policy_class(config: TrainerConfigDict) -> Optional[Type[Policy]]:
    """Policy class picker function. Class is chosen based on DL-framework.
    Args:
        config (TrainerConfigDict): The trainer's configuration dict.
    Returns:
        Optional[Type[Policy]]: The Policy class to use with PPOTrainer.
            If None, use `default_policy` provided in build_trainer().
    """
    return AgentPolicy


load_envs(os.getcwd()) # Load envs
load_models(os.getcwd()) # Load models


AgentTrainer = PPOTrainer.with_updates(
        default_policy=AgentPolicy,
        get_policy_class=get_policy_class)

sub_dir = "/test_run" # test_run
analysis = tune.run(AgentTrainer, 
                    local_dir=result_dir+sub_dir, 
                    name=exp_name, 
                    stop={'timesteps_total': max_timestep}, 
                    resume=resume,
                    config=experiments, 
                    keep_checkpoints_num=3, 
                    checkpoint_freq=10, 
                    checkpoint_score_attr="training_iteration", 
                    checkpoint_at_end=True)
