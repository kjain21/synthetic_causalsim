import os
import numpy as np
import env
from param import config


def load_chunk_sizes():
    # bytes of video chunk file at different bitrates

    # source video: "Envivio-Dash3" video H.264/MPEG-4 codec
    # at bitrates in {300,750,1200,1850,2850,4300} kbps

    # original video file:
    # https://github.com/hongzimao/pensieve/tree/master/video_server

    # download video size folder if not existed
    video_folder = env.__path__[0] + '/videos/'
    os.makedirs(video_folder, exist_ok=True)
    if not os.path.exists(video_folder + 'video_sizes.npy'):
        wget.download(
            'https://www.dropbox.com/s/hg8k8qq366y3u0d/video_sizes.npy?dl=1',
            out=video_folder + 'video_sizes.npy')

    chunk_sizes = np.load(video_folder + 'video_sizes.npy')

    return chunk_sizes


def load_traces():
    """
    :type seed: int
    :rtype: (list of (np.ndarray | list, np.ndarray | list), np.ndarray)
    """
    return np.load('env/white_inputs.npy')


def sample_trace(np_random):
    # sample a trace
    trace_idx = np_random.choice(5000)
    # sample a starting point
    init_t_idx = np_random.choice(489)
    # return a trace and the starting t
    return trace_idx, init_t_idx
