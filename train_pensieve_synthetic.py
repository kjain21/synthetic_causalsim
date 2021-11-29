from core_pg import sample_action, train_actor_critic
from core_log import log_a2c
from env.abr import ABRSimEnv
from nn import PensNet
from pensieve_utils import ewma
from buffer import TransitionBuffer
from torch.utils.tensorboard import SummaryWriter
from normer import Normer
import torch
import os
import numpy as np
import time
from tqdm.notebook import trange, tqdm
from param import config
from env.model import *


def train_pensieve(gamma, ent_max, ent_decay, name):
    # Create some folders to save models and tensorboard data
    os.makedirs(f'models/{name}/', exist_ok=True)
    os.makedirs('tensorboard/', exist_ok=True)

    # Monitor training with tensorboard
    monitor = SummaryWriter(f'tensorboard/{name}', flush_secs=10)
    
    # An object to normalize the observation vectors, This usually speeds up training.
    normer = Normer()

    # A set of hyperparameters
    # DO NOT CHANGE
    LR = 1e-2                       #  Learning rate
    WD = 1e-4                       #  Weight decay (or in other words, L2 regularization penalty)
    NUM_EPOCHS = 750                #  How many epochs to train
    EPOCH_SAVE = 100                #  How many epochs till we save the model
    ENT_MAX = ent_max               #  Initial value for entropy
    ENT_DECAY = ent_decay           #  Entropy decay rate
    REW_SCALE = 25                  #  Reward scale
    LAMBDA = 0.95                   #  Lambda, used for GAE-style advantage calculation
    GAMMA = gamma                   #  Gamma in discounted rewards
    
    # We will save episodic returns for comparison later
    returns = np.zeros(NUM_EPOCHS)

    # Making runs deterministic by using specific random seeds
    torch.random.manual_seed(123)
    np.random.seed(123)
    
    # The ABR environment, the argument is the random seed
    env = ABRSimEnv()

    print("gets here")

    obs, _ = env.reset()
    
    # This is the width of the observation
    obs_len = obs.shape[0]
    # This is the number of possible actions
    act_len = env.action_space.n

    # Entropy factor, this is the entity pushing for exploration
    entropy_factor = ENT_MAX

    # The actor network, which we call policy_net here
    policy_net = torch.jit.script(PensNet(obs_len, act_len, [32, 16]))
    # The critic network, which we call value_net here
    value_net = torch.jit.script(PensNet(obs_len, 1, [32, 16]))
    # A buffer that takes care of storing interaction data. This replaces the 3 lists we used for the tabular policy gradient assignment
    buff = TransitionBuffer(obs_len, env.total_num_chunks)

    # five_state_dict = policy_net.state_dict()
    # six_state_dict = policy_net.state_dict()

    # Optimizers that apply gradients, SGD could be used instead but this optimizer performs better
    net_opt_p = torch.optim.Adam(policy_net.parameters(), lr=LR, weight_decay=WD)
    net_opt_v = torch.optim.Adam(value_net.parameters(), lr=LR, weight_decay=WD)
    
    # The loss for training the critic, which is basically a Mean Squared Error loss
    net_loss = torch.nn.MSELoss(reduction='mean')

    # Check elapsed time
    last_time = time.time()

    # Training process
    for epoch in trange(NUM_EPOCHS):

        # Same as before, each epoch is one episode
        obs, _ = env.reset()
        buff.reset_head()
        done = False

        while not done:
            # We sample an action
            act = sample_action(policy_net, normer(obs), torch.device('cpu'))

            # We take a step
            next_obs, rew, done, info = env.step(act)
            # print(rew)

            # Save our interactions in the buffer
            buff.add_exp(obs, act, rew, next_obs, done, info['stall_time'])

            obs = next_obs

        # The buffer size is set to the length of episodes (the video length), so this line is a sanity check
        assert buff.buffer_full()

        # Get all the saved interactions from the buffer manager
        all_states, all_next_states, all_actions_np, all_rewards, all_dones = buff.get()
        # print(all_rewards)

        # Train A2C with GAE and entropy regularizer, the names sound scary but they are simpler than you think
        pg_loss, v_loss, real_entropy, ret_np, v_np, adv_np = train_actor_critic(value_net, policy_net, net_opt_p, net_opt_v, net_loss, torch.device('cpu'), 
                                                                                 all_actions_np, normer(all_next_states), all_rewards / REW_SCALE, 
                                                                                 normer(all_states), all_dones, entropy_factor, GAMMA, LAMBDA)

        # Normalized entropy, it ranges from 1 (fully random policy) to 0 (One action is deterministically taken)
        norm_entropy = real_entropy / - np.log(act_len)
        
        # Decay the entropy factor
        entropy_factor = max(0, entropy_factor-ENT_DECAY)

        # Save the model
        if epoch % EPOCH_SAVE == 0:
            state_dicts = [policy_net.state_dict(), value_net.state_dict()]
            torch.save(state_dicts, f'models/{name}/model_{epoch}')

        # Update elapsed time
        curr_time = time.time()
        elapsed = curr_time - last_time
        last_time = curr_time

        # Log the training via tensorboard
        log_a2c(buff, ret_np, v_np, adv_np, pg_loss, v_loss, entropy_factor, norm_entropy, elapsed, monitor, epoch)
        
        # Save returns
        returns[epoch] = buff.reward_fifo.sum()

    # Save the final model
    state_dicts = [policy_net.state_dict(), value_net.state_dict()]
    state_dict_policy = policy_net.state_dict()
    torch.save(state_dicts, f'models/{name}/model_{epoch}')
    # torch.save(policy_net.state_dict(), f'models/only_policy/model_{epoch}')
    print(returns)
    return returns


if __name__ == '__main__':
    train_pensieve(gamma=0.9, ent_max=1, ent_decay=1/400, name='synthetic_test')
