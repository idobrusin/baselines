from baselines.common.vec_env import VecEnvWrapper
import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        if isinstance(venv.observation_space, dict):
            wos = venv.observation_space['img'] # wrapped ob space
            self.nchannel = wos.low.shape[-1]
            low = np.repeat(wos.low, self.nstack, axis=-1)
            high = np.repeat(wos.high, self.nstack, axis=-1)
            self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
            observation_space = venv.observation_space
            observation_space['img'] = \
                spaces.Box(low=low, high=high, dtype=venv.observation_space['img'].dtype)
        else:
            wos = venv.observation_space # wrapped ob space
            low = np.repeat(wos.low, self.nstack, axis=-1)
            high = np.repeat(wos.high, self.nstack, axis=-1)
            self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
            observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-self.nchannel, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        if isinstance(obs, dict):
            self.stackedobs[..., -self.nchannel:] = obs['img']
            obs['img'] = self.stackedobs
            return obs, rews, news, infos
        else:
            self.stackedobs[..., -self.nchannel:] = obs
            return self.stackedobs, rews, news, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        if isinstance(obs, dict):
            self.stackedobs[..., -self.nchannel:] = obs['img']
            obs['img'] = self.stackedobs
            return obs
        else:
            self.stackedobs[..., -self.nchannel:] = obs
            return self.stackedobs

    def close(self):
        self.venv.close()