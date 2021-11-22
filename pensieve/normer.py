import numpy as np
from sklearn.preprocessing import Normalizer
import pickle
from abr_env.abr import ABRSimEnv
from buffer import TransitionBuffer
from abr_env.model import *


def create_normer():
    transformer = Normalizer()
    np.random.seed(123)
    env = ABRSimEnv()

    obs, _ = env.reset()

    obs_len = obs.shape[0]
    act_len = env.action_space.n
    buff = TransitionBuffer(obs_len, env.total_num_chunks*100)

    buff.reset_head()
    done = True
    t = 0

    while not buff.buffer_full():
        if done:
            obs, obs_ext = env.reset()

        act = np.random.choice(act_len)

        next_obs, rew, done, info = env.step(act)
        print(f'At chunk {t}, the agent took action {act}, and got a reward {rew}')
        print(f'\t\tThe observation was {obs}')

        buff.add_exp(obs, act, rew, next_obs, done, info['stall_time'])

        # next state
        obs = next_obs
        t+=1

    assert buff.buffer_full()

    all_states, all_next_states, all_actions_np, all_rewards, all_dones = buff.get()

    transformer.fit_transform(all_states)

    with open('normer1.pkl', 'wb') as fandle:
        pickle.dump(transformer, fandle)


class Normer(object):
    def __init__(self):
        with open('pensieve/normer1.pkl', 'rb') as fandle:
            self.transformer = pickle.load(fandle)
    
    def __call__(self, data):
        return self._normit(data)
        
    def _normit(self, data):
        if data.ndim == 1:
            resqueeze = True
            data = data.reshape(1, -1)
        else:
            resqueeze = False
        out = self.transformer.transform(data)
        if resqueeze:
            out = out[0]
        return out

if __name__ == '__main__':
    create_normer()