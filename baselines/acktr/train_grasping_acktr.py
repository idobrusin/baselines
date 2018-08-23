#!/usr/bin/env python3

import tensorflow as tf
from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import set_global_seeds
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.bench import Monitor
import gym
import datetime
import os.path as osp
from gym_grasping.envs.grasping_env import GraspingEnv

def train(env_id, num_timesteps, seed):
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=200,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=True)

        env.close()

def main():
    logger.configure(dir=osp.join("/home/hermannl/master_thesis/git/baselines_private/baselines/acktr/logs/",
                                  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                     format_strs=["tensorboard", "stdout", "csv", "log"])
    train(env_id='kuka_block_cont-v0', num_timesteps=10e7, seed=10)

if __name__ == "__main__":
    main()
