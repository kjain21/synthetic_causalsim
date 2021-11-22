# import numpy as np
# from collections import deque
# from pensieve.abr_env import box, discrete, logger
# from pensieve.abr_env.thr_calc import thr_discrete, thr_slow_start, thr_integrate
# from pensieve.abr_env.trace_master import TraceMasterGenerative
# from pensieve.abr_env.param import config
# from pensieve.abr_env.trace_loader import load_traces, load_chunk_sizes, sample_trace, get_chunk_time
import numpy as np
from collections import deque
import torch
from abr_env import box, discrete, logger
from abr_env.param import config
from abr_env.trace_loader import load_chunk_sizes, sample_trace, load_traces


class ABRSimEnv(object):
    """
    Adapt bitrate during a video playback with varying network conditions.
    The objective is to (1) reduce stall (2) increase video quality and
    (3) reduce switching between bitrate levels. Ideally, we would want to
    *simultaneously* optimize the objectives in all dimensions.

    * STATE *
        [The throughput estimation of the past chunk (chunk size / elapsed time),
        download time (i.e., elapsed time since last action), current buffer ahead,
        number of the chunks until the end, the bitrate choice for the past chunk,
        current chunk size of bitrate 1, chunk size of bitrate 2,
        ..., chunk size of bitrate 5]

        Note: we need the selected bitrate for the past chunk because reward has
        a term for bitrate change, a fully observable MDP needs the bitrate for past chunk

    * ACTIONS *
        Which bitrate to choose for the current chunk, represented as an integer in [0, 4]

    * REWARD *
        At current time t, the selected bitrate is b_t, the stall time between
        t to t + 1 is s_t, then the reward r_t is
        b_{t} - 4.3 * s_{t} - |b_t - b_{t-1}|
        Note: there are different definitions of combining multiple objectives in the reward,
        check Section 5.1 of the first reference below.

    * REFERENCE *
        Section 4.2, Section 5.1
        Neural Adaptive Video Streaming with Pensieve
        H Mao, R Netravali, M Alizadeh
        https://dl.acm.org/citation.cfm?id=3098843

        A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP
        X Yin, A Jindal, V Sekar, B Sinopoli
        https://dl.acm.org/citation.cfm?id=2787486
    """

    def __init__(self):
        # observation and action space
        self.trace_idx = None
        self.rtt = None
        self.curr_t_idx = None
        self.chunk_idx = None
        self.buffer_size = None
        self.past_action = None
        self.past_chunk_throughputs = None
        self.past_chunk_download_times = None
        self.np_random = None
        self.obs_high = None
        self.obs_low = None
        self.observation_space = None
        self.action_space = None
        self.setup_space()
        # set up seed
        self.seed(config.seed)
        # load all video chunk sizes
        self.chunk_sizes = load_chunk_sizes()
        # mapping between action and bitrate level
        self.bitrate_map = [0.3, 0.75, 1.2, 1.85, 2.85, 4.3]  # Mbps
        # how many past throughput to report
        self.past_chunk_len = 8
        self.rebuf_penalty = 4.3
        self.data = load_traces()
        self.extractor = torch.load('abr_env/extractor.pth')
        self.predictor = torch.load('abr_env/predictor.pth')
        self.buffer_mean = np.load('abr_env/buffer_mean.npy')
        self.next_buffer_mean = np.load('abr_env/next_buffer_mean.npy')
        self.buffer_std = np.load('abr_env/buffer_std.npy')
        self.next_buffer_std = np.load('abr_env/next_buffer_std.npy')
        self.chosen_chunk_size_mean = np.load('abr_env/chosen_chunk_size_mean.npy')
        self.chosen_chunk_size_std = np.load('abr_env/chosen_chunk_size_std.npy')
        self.download_time_mean = np.load('abr_env/download_time_mean.npy')
        self.download_time_std = np.load('abr_env/download_time_std.npy')
        # assert number of chunks for different bitrates are all the same
        assert len(np.unique([len(chunk_size) for chunk_size in self.chunk_sizes])) == 1
        self.total_num_chunks = len(self.chunk_sizes[0])

    def observe(self):
        if self.chunk_idx < self.total_num_chunks:
            valid_chunk_idx = self.chunk_idx
        else:
            valid_chunk_idx = 0

        if self.past_action is not None:
            valid_past_action = self.past_action
        else:
            valid_past_action = 0

        # network throughput of past chunk, past chunk download time,
        # current buffer, number of chunks left and the last bitrate choice
        obs_arr = [self.past_chunk_throughputs[-i] for i in range(config.mpc_lookback, 0, -1)]
        obs_arr.extend([self.past_chunk_download_times[-i] for i in range(config.mpc_lookback, 0, -1)])
        obs_arr.extend([self.buffer_size, self.total_num_chunks - self.chunk_idx, valid_past_action])

        # current chunk size of different bitrates
        for chunk_idx_add in range(valid_chunk_idx, config.mpc_lookahead + valid_chunk_idx):
            obs_arr.extend(self.chunk_sizes[i][chunk_idx_add % self.total_num_chunks] for i in range(6))

        # fit in observation space
        for i in range(len(obs_arr)):
            if obs_arr[i] > self.obs_high[i]:
                logger.warn('Observation at index ' + str(i) +
                            ' at chunk index ' + str(self.chunk_idx) +
                            ' has value ' + str(obs_arr[i]) +
                            ', which is larger than obs_high ' +
                            str(self.obs_high[i]))
                obs_arr[i] = self.obs_high[i]

        obs_arr = np.array(obs_arr)
        assert self.observation_space.contains(obs_arr), f"The violating observation is {obs_arr}"

        return obs_arr[:2 * config.mpc_lookback + 3 + 6], obs_arr

    def reset(self, trace_choice=None):
        if trace_choice is None:
            self.trace_idx, self.curr_t_idx = sample_trace(self.np_random)
        else:
            assert trace_choice[0] < 5000
            assert trace_choice[1] < 489
            self.trace_idx = trace_choice[0]
            self.curr_t_idx = trace_choice[1]
        self.chunk_idx = 0
        self.buffer_size = 0.0  # initial download time not counted
        self.past_action = None
        self.past_chunk_throughputs = deque(maxlen=self.past_chunk_len)
        self.past_chunk_download_times = deque(maxlen=self.past_chunk_len)
        for _ in range(self.past_chunk_len):
            self.past_chunk_throughputs.append(0)
            self.past_chunk_download_times.append(0)

        return self.observe()

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0] * (3 + 2 * config.mpc_lookback + 6 * config.mpc_lookahead))
        self.obs_high = np.array([100e6] * config.mpc_lookback + [5000] * config.mpc_lookback + [100, 500, 5] +
                                 [10e6] * (6 * config.mpc_lookahead))
        self.observation_space = box.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = discrete.Discrete(6)

    def step(self, action):
        assert self.action_space.contains(action)

        # Note: sizes are in bytes, times are in seconds
        chunk_size = self.chunk_sizes[action][self.chunk_idx]

        # compute chunk download time based on trace
        white_buffer = self.buffer_size - self.buffer_mean
        white_buffer /= self.buffer_std
        white_chunk_size = chunk_size - self.chosen_chunk_size_mean
        white_chunk_size /= self.chosen_chunk_size_std
        chosen_chunk_size, min_rtt, c_hat = self.data[get_index_number(self.trace_idx, self.curr_t_idx)]
        extractor_input = np.array([chosen_chunk_size, min_rtt, c_hat])
        extractor_input = np.expand_dims(extractor_input, axis=0)
        extractor_input_tensor = torch.as_tensor(extractor_input, dtype=torch.float32, device=torch.device('cpu'))
        with torch.no_grad():
            feature_tensor = self.extractor(extractor_input_tensor)
        predictor_input = np.array([white_buffer, white_chunk_size, min_rtt])
        predictor_input = np.expand_dims(predictor_input, axis=0)
        predictor_input_tensor = torch.as_tensor(predictor_input, dtype=torch.float32, device=torch.device('cpu'))
        predictor_input_tensor = torch.cat((predictor_input_tensor, feature_tensor), dim=1)
        with torch.no_grad():
            prediction = self.predictor(predictor_input_tensor)
        next_buffer_white_tensor, down_time_white_tensor = prediction[0, 0], prediction[0, 1]
        next_buffer_white = next_buffer_white_tensor.cpu().numpy()
        down_time_white = down_time_white_tensor.cpu().numpy()
        download_time = (down_time_white * self.download_time_std) + self.download_time_mean
        next_buffer = (next_buffer_white * self.next_buffer_std) + self.next_buffer_mean
        self.curr_t_idx += 1
        if self.curr_t_idx >= 489:
            self.curr_t_idx = 0
        # compute buffer size
        rebuffer_time = max(download_time - self.buffer_size, 0)

        # update video buffer
        self.buffer_size = next_buffer

        # bitrate change penalty
        if self.past_action is None:
            bitrate_change = 0
        else:
            bitrate_change = np.abs(self.bitrate_map[action] - self.bitrate_map[self.past_action])

        # linear reward
        # (https://dl.acm.org/citation.cfm?id=3098843 section 5.1, QoE metrics (1))
        reward = self.bitrate_map[action] - self.rebuf_penalty * rebuffer_time - bitrate_change

        # store action for future bitrate change penalty
        self.past_action = action

        # update observed network bandwidth and duration
        self.past_chunk_throughputs.append(self.chunk_sizes[action][self.chunk_idx] / float(download_time))
        self.past_chunk_download_times.append(download_time)

        # advance video
        self.chunk_idx += 1
        done = (self.chunk_idx == self.total_num_chunks)
        obs, obs_extended = self.observe()

        return obs, reward, done, \
               {
                   'bitrate': self.bitrate_map[action],
                   'stall_time': rebuffer_time,
                   'bitrate_change': bitrate_change,
                   'obs_extended': obs_extended
               }


def get_index_number(trace_idx, time_idx):
    return trace_idx * 489 + time_idx
