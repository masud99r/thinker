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
from generator import StarGAN
from os.path import join, exists
import os 
from os import mkdir
import random
import argparse

import numpy as np
import copy

from torch.autograd import Variable



ray.init(num_cpus=20, num_gpus=1) # Only call this once.

# Fix numeric divergence due to bug in Cudnn # need to check

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)
# device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--algo", default='thinker', help="RL algorithm to use")
parser.add_argument("--env", default='maze', help="Procgen env to use")
parser.add_argument("--seed", type=int, default=100, help="Seed for this run")
parser.add_argument("--epoch", type=int, default=0, help="Whether load trained generator. 0 = False, 1 = True")
parser.add_argument("--resume", type=int, default=0, help="Whether resume RL training, considering the cluster and generators are trained. 0 = False, 1 = True, Note: Not thoroughly tested.")
parser.add_argument("--n_cluster", type=int, default=3, help="Number of cluster")
parser.add_argument("--n_epochs", type=int, default=5, help="Total training epoch of generator")
parser.add_argument("--max_timestep", type=int, default=25000000, help="Timesteps to run the algorithms")
parser.add_argument("--result_dir", default='./', help="directory to save results")

opt = parser.parse_args()
print(opt)

algo = opt.algo
epoch = opt.epoch
n_epochs=opt.n_epochs
n_cluster = opt.n_cluster
max_timestep = opt.max_timestep


if opt.resume == 1:
    resume = True
else:
    resume = False

# if opt.load_trained == 1:
#     load_trained = True
# else:
#     load_trained = False
load_trained = False
if epoch > 0:
    load_trained = True

if resume: # override for resume, generator must be trained before this run
    load_trained = True 
# else:
#     load_trained = False

# if load_trained:
#     load_epoch = n_epochs - 1 # last epoch
# else:
#     load_epoch = 0
# load_trained = True

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

if algo=='thinker':
    gdir = env_name+"_gen_"+str(n_cluster)
    # gdir = env_name+'_'+str(num_levels)+'_'+str(seed)
    thinker_dir = result_dir+"/"+gdir
    if not exists(thinker_dir):
        mkdir(thinker_dir)
    cluster_dir = join(thinker_dir+'/cluster/')
    gen_dir = join(thinker_dir+'/generator/')
    if not exists(cluster_dir):
        mkdir(cluster_dir)
    if not exists(gen_dir):
        mkdir(gen_dir)
    cluster_model = Cluster(cluster_dir=cluster_dir,load_model=load_trained, n_components=n_cluster, device=device)

def train_clustering(data):
    cluster_model.train(data)

selected_attrs = list(range(0, n_cluster))
generator = StarGAN(gan_dir=gen_dir, n_epochs=n_epochs, epoch=epoch, batch_size=16, n_cpu=3, img_height=64, img_width=64, channels=3, sample_interval=10, checkpoint_interval=1, selected_attrs=selected_attrs, device=device)
    # generators_dict = {}

def train_generator(data):
    if load_trained:
        generator.load_generator()
    else:
        class_labels = cluster_model.get_cluster_labels(data)
        label = [[0] * n_cluster] * len(class_labels)
        labels = []
        _label = [0] * n_cluster
        for i in range(len(class_labels)):
            label = copy.deepcopy(_label)
            label[class_labels[i]] = 1.0 # other class set to 0
            labels.append(label)
        
        if generator.train_count == 0:
            generator.init_generator()
        
        generator.train(data, labels, n_iteration=opt.n_epochs)

# save in global trajectory_obs 
trajectory_obs = None
def convert_obs(obs):
    global trajectory_obs
    # print('obs.shape', obs.shape)
    global count_dist_train
    
    if device == 'cpu':
        obs = obs.type(torch.FloatTensor)
    else: 
        obs = obs.type(torch.cuda.FloatTensor)
    
    if trajectory_obs is None: # first time
        trajectory_obs = obs
    elif trajectory_obs.shape[0] < 5000: # convert when enough data to train generator, 35 is to skipp the first batch, the next RL iteration will have the replay buffer sized obs
        # append obs to trajctory
        trajectory_obs = torch.cat((trajectory_obs, obs), 0)
    else:
        if generator.train_count == 0: # train once, might want to train multiple times
            train_clustering(trajectory_obs) # in case of resume it will load the latest trained model
            train_generator(trajectory_obs) # in case of resume it will load the latest trained model

    if generator.train_count == 0: # not trained yet
        return obs
    else:    
        # obs = obs.type(torch.FloatTensor)
        source_labels = cluster_model.get_labels(obs)
        target_labels = np.random.randint(n_cluster, size=len(source_labels))
        for j in range(len(target_labels)):
            if source_labels[j] == target_labels[j]:
                target_labels[j] = (target_labels[j] + 1) % n_cluster
        labels = []
        for k in range(len(target_labels)):
            label = [0.0] * n_cluster
            label[target_labels[k]] = 1
            labels.append(label)
        # print('source_labels', source_labels)
        # print('target labels', labels)
        # labels = torch.stack(labels, dim=0)
        # labels = Variable(labels.type(Tensor))
        # labels = torch.Tensor
        if device == 'cpu':
            labels = torch.FloatTensor(labels)
        else: 
            labels = torch.cuda.FloatTensor(labels)
        # print('obs convert', obs)
        # print('labels convert', labels)

        obs = generator.convert_dist(obs, labels)
        # obs = obs.squeeze(0).detach()
       
    return obs

# def thinker_loss(policy, model, dist_class, train_batch):
#     loss1 = ppo_surrogate_loss(policy, model, dist_class, train_batch)
#     # change observation (current), train_batch
#     obs = convert_obs(train_batch[SampleBatch.CUR_OBS])
#     train_batch[SampleBatch.CUR_OBS] = obs #.to(device)
#     loss2 = ppo_surrogate_loss(policy, model, dist_class, train_batch) # counterfactual
    
#     loss = (loss1 + loss2)/2
#     return loss

def thinker_loss(policy, model, dist_class, train_batch):
    obs = convert_obs(train_batch[SampleBatch.CUR_OBS])
    train_batch[SampleBatch.CUR_OBS] = obs #.to(device)
    loss = ppo_surrogate_loss(policy, model, dist_class, train_batch) # counterfactual
    return loss
ThinkerPolicy = PPOTorchPolicy.with_updates(
    name="ThinkerPolicy",
    loss_fn=thinker_loss)

def get_policy_class(config: TrainerConfigDict) -> Optional[Type[Policy]]:
    """Policy class picker function. Class is chosen based on DL-framework.
    Args:
        config (TrainerConfigDict): The trainer's configuration dict.
    Returns:
        Optional[Type[Policy]]: The Policy class to use with PPOTrainer.
            If None, use `default_policy` provided in build_trainer().
    """
    return ThinkerPolicy


load_envs(os.getcwd()) # Load envs
load_models(os.getcwd()) # Load models

ThinkerTrainer = PPOTrainer.with_updates(
        default_policy=ThinkerPolicy,
        get_policy_class=get_policy_class)

sub_dir = "/test_run" # run_seeds
analysis = tune.run(ThinkerTrainer, 
                    local_dir=result_dir+sub_dir, 
                    name=exp_name, 
                    stop={'timesteps_total': max_timestep}, 
                    resume=resume,
                    config=experiments, 
                    keep_checkpoints_num=3, 
                    checkpoint_freq=10, 
                    checkpoint_score_attr="training_iteration", 
                    checkpoint_at_end=True)
