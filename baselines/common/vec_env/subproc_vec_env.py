import numpy as np
from multiprocessing import Process, Pipe
from . import VecEnv, CloudpickleWrapper
import time

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'step_with_curriculum_reset':
                action = data[0]
                data = data[1]
                ob, reward, done, info = env.step(action)
                if done:
                    ob = env.reset(data=data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'reset_from_curriculum':
                ob = env.reset(data=data)
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        self.env_fns = env_fns
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.remotes = list(self.remotes)
        self.work_remotes = list(self.work_remotes)
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), self.observation_space, self.action_space)

    def restart_envs(self, poll_results, do_step=False):
        for i, r in enumerate(poll_results):
            if not r:
                self.ps[i].terminate()
                time.sleep(10)
                self.remotes[i], self.work_remotes[i] = Pipe()
                self.ps[i] = Process(target=worker, args=(self.work_remotes[i], self.remotes[i], CloudpickleWrapper(self.env_fns[i])))
                self.ps[i].daemon = True
                self.ps[i].start()
                self.work_remotes[i].close()
                self.remotes[i].send(('reset', None))
                if do_step:
                    self.remotes[i].recv()
                    self.remotes[i].send(('step', self.action_space.sample()))

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_async_with_curriculum_reset(self, actions, data):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step_with_curriculum_reset', (action, data)))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        remotes_responsive = [remote.poll(20) for remote in self.remotes]
        while not np.all(remotes_responsive):
            try:
                print(remotes_responsive)
                print("restart envs")
                self.restart_envs(remotes_responsive, do_step=True)
                remotes_responsive = [remote.poll(20) for remote in self.remotes]
            except Exception:
                print("exception")
                continue
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        if isinstance(obs[0], dict):
            return {key: np.stack([ob[key] for ob in obs]) for key in obs[0].keys()}, np.stack(rews), np.stack(dones), infos
        else:
            return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        remotes_responsive = [remote.poll(20) for remote in self.remotes]
        while not np.all(remotes_responsive):
            try:
                print(remotes_responsive)
                print("restart envs")
                self.restart_envs(remotes_responsive)
                remotes_responsive = [remote.poll(20) for remote in self.remotes]
            except Exception:
                print("exception")
                continue
        obs = np.stack([remote.recv() for remote in self.remotes])
        if isinstance(obs[0], dict):
            return {key: np.stack([ob[key] for ob in obs]) for key in obs[0].keys()}
        else:
            return np.stack(obs)

    def reset_from_curriculum(self, data):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset_from_curriculum', data))
        remotes_responsive = [remote.poll(20) for remote in self.remotes]
        while not np.all(remotes_responsive):
            try:
                print(remotes_responsive)
                print("restart envs")
                self.restart_envs(remotes_responsive)
                remotes_responsive = [remote.poll(20) for remote in self.remotes]
            except Exception:
                print("exception")
                continue
        obs = np.stack([remote.recv() for remote in self.remotes])
        if isinstance(obs[0], dict):
            return {key: np.stack([ob[key] for ob in obs]) for key in obs[0].keys()}
        else:
            return np.stack(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
