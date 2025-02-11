import numpy as np
from sklearn.preprocessing import Normalizer
import pickle
from env.abr import ABRSimEnv
from buffer import TransitionBuffer
from env.model import *


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
        # with open('pensieve/normer_param_update.pkl', 'rb') as fandle:
        #     self.transformer = pickle.load(fandle)
        pass
    
    def __call__(self, data):
        return self._normit(data)
        
    def _normit(self, data):
        if data.ndim == 1:
            resqueeze = True
            data = data.reshape(1, -1)
        else:
            resqueeze = False

        # out = self.transformer.transform(data)

        assert data.shape[1] == 19
        out = np.array(data)
        out[:, :5] = data[:, :5] * 8 / 1e6
        out[:, 10] = data[:, 10] / 40 * 2 - 1
        out[:, 11] = data[:, 11] / 490 * 2 - 1
        out[:, 12] = data[:, 12] / 3 - 1
        out[:, 13:19] = data[:, 13:19] * 8 / 1e6

        if resqueeze:
            out = out[0]
        return out

if __name__ == '__main__':
    # create_normer()
    normer = Normer()