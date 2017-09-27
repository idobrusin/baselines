#!/usr/bin/env python
import argparse
import logging
import os
import tensorflow as tf
import gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
import datetime
import os.path as osp

def train(env_id, num_timesteps, seed):
    env=gym.make(env_id)
    logger.configure(dir = osp.join("logs/",
        datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")))
    if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    set_global_seeds(seed)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    with tf.Session(config=tf.ConfigProto()) as session:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        print("observation_space",ob_dim)
        print("action_space", ac_dim)
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=2500,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=True)

        env.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
    parser.add_argument('--env_id', type=str, default="Jaco-v1")
    args = parser.parse_args()
    train(args.env_id, num_timesteps=1e8, seed=1)
