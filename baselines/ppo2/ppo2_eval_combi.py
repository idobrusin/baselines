import sys
import os.path as osp
import multiprocessing
import cloudpickle
import tensorflow as tf

from baselines.ppo2.policies import *
from baselines.ppo2.ppo2 import Model, Runner
from baselines.common import set_global_seeds

def eval(env, nsteps, gamma=0.99, lam=0.95, load_path=None, nminibatches=4, vf_coef=0.5,  max_grad_norm=0.5, ent_coef=.01, seed=None) :
    set_global_seeds(seed)
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    #policy = CnnPolicy
    #policy = MlpPolicy

    policy = AsyncCombiPolicy

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    model = make_model()
    model.load(load_path)
    env.load_norm(load_path)

    nsteps = 1
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    return runner

