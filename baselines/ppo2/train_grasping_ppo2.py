#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
import datetime
import os.path as osp
def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from gym_grasping.envs.GraspingCamGymEnv import GraspingCamGymEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=4096, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=1e-4, # changed
        cliprange=0.2,
        total_timesteps=num_timesteps)
    # try: noptepochs 2 /4
    #
    #
    #
    #


def main():
    logger.configure(dir=osp.join("/home/hermannl/master_thesis/git/baselines_private/baselines/ppo2/logs/",
                                    datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                     format_strs=["tensorboard", "stdout", "csv", "log"])
    train(env_id='PR2_Cube_cont-v0', num_timesteps=10e8, seed=10)


if __name__ == '__main__':
    main()
