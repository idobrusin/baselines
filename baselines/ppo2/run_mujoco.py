#!/usr/bin/env python
import argparse
from baselines import bench, logger
import os.path as osp
import datetime

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    logger.configure(dir = osp.join("/home/hermannl/master_project/git/baselines/baselines/ppo2/logs/", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),format_strs=["tensorboard", "stdout", "csv", "log"])
    def make_env():
        env = gym.make(env_id)
        env.seed(seed)
        env = bench.Monitor(env, logger.get_dir(),allow_early_resets=True)
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=8192, nminibatches=32,
        lam=0.97, gamma=0.995, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=5e-5,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='JacoPush-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e8))
    args = parser.parse_args()
    logger.configure(format_strs=["tensorboard", "stdout", "csv", "log"])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
