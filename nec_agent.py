import random
import torch

import torch.optim as optim
import scipy.signal as signal

from collections import namedtuple
from dnd import DND
from itertools import count
from utils.replay_memory import ReplayMemory
from torch.autograd import Variable

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

use_cuda = torch.cuda.is_available()

def Tensor(nparray):
  if use_cuda:
    return torch.Tensor(nparray).cuda()
  else:
    return torch.Tensor(nparray)

def discount(x, gamma):
  """
  Compute discounted sum of future values
  out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
  """
  return signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def inverse_distance(h, h_i, epsilon=1e-3):
  return 1 / (torch.dist(h, h_i) + epsilon)

def epsilon_schedule(EPS_END=0.05, EPS_START=0.95, EPS_DECAY=200):
  def schedule(t):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)
  return schedule

class NECAgent:
  def __init__(self,
               env,
               embedding_network,
               replay_memory=ReplayMemory(100000),
               epsilon_schedule=epsilon_schedule,
               batch_size=32,
               sgd_learning_rate=1e-2,
               q_learning_rate=0.5,
               gamma=0.99,
               lookahead_horizon=100,
               update_period=4,
               kernel=inverse_distance,
               num_neighbors=50,
               max_memory=500000,
               warmup_period=100,
               test_period=None):
    """
    Instantiate an NEC Agent

    Parameters
    ----------
    env: gym.Env
      gym environment to train on
    embedding_network: torch.nn.Module
      Model to extract the embedding from a state
    replay_memory: ReplayMemory
      Replay memory to sample from for embedding network updates
    epsilon_schedule: (int) => (float)
      Function that determines the epsilon for epsilon-greedy exploration from the timestep t
    batch_size: int
      Batch size to sample from the replay memory
    sgd_learning_rate: float
      Learning rate to use for RMSProp updates to the embedding network
    q_learning_rate: float
      Learning rate to use for Q-updates on DND updates
    gamma: float
      Discount factor
    lookahead_horizon: int
      Lookahead horizon to use for N-step Q-value estimates
    update_period: int
      Inverse of rate at which embedding network gets updated
      i.e. if 1 then update after every timestep, if 16 then update every 16 timesteps, etc.
    kernel: (torch.autograd.Variable, torch.autograd.Variable) => (torch.autograd.Variable)
      Kernel function to use for DND lookups
    num_neighbors: int
      Number of neighbors to return in K-NN lookups in DND
    max_memory: int
      Maximum number of key-value pairs to store in DND
    warmup_period: int
      Number of timesteps to act randomly before learning
    test_period: int
      Number of episodes between each test iteration
    """

    self.env = env
    self.embedding_network = embedding_network
    if use_cuda:
      self.embedding_network.cuda()

    self.replay_memory = replay_memory
    self.epsilon_schedule = epsilon_schedule
    self.batch_size = batch_size
    self.q_learning_rate = q_learning_rate
    self.gamma = gamma
    self.lookahead_horizon = lookahead_horizon
    self.update_period = update_period
    self.warmup_period = warmup_period
    self.test_period = test_period

    self.transition_queue = []
    self.optimizer = optim.RMSprop(self.embedding_network.parameters(), lr=sgd_learning_rate)

    state_dict = self.embedding_network.state_dict()
    self.dnd_list = [DND(kernel, num_neighbors, max_memory, state_dict[next(reversed(state_dict))].size()[0]) for _ in range(env.action_space.n)]

  def choose_action(self, state_embedding, epsilon):
    """
    Choose action from epsilon-greedy policy
    If not a random action, choose the action that maximizes the Q-value estimate from the DNDs
    """
    if random.uniform(0, 1) < epsilon or self.before_learning:
      return random.randint(0, self.env.action_space.n - 1)
    else:
      q_estimates = [dnd.lookup(state_embedding) for dnd in self.dnd_list]
      return torch.cat(q_estimates).max(0)[1].data[0]

  def Q_lookahead(self, t):
    """
    Return the N-step Q-value lookahead from time t in the transition queue
    """
    x = [transition.reward for transition in self.transition_queue[t:t+self.lookahead_horizon]]
    lookahead = discount(x, self.gamma)[0]
    if len(self.transition_queue) > t + self.lookahead_horizon and not self.before_learning:
      state = self.transition_queue[t+self.lookahead_horizon-1].state
      state_embedding = self.embedding_network(Variable(Tensor(state)).unsqueeze(0))
      return self.gamma ** self.lookahead_horizon * torch.cat([dnd.lookup(state_embedding) for dnd in self.dnd_list]).max() + lookahead
    else:
      return Variable(Tensor([lookahead]))

  def Q_update(self, q_initial, q_n):
    """
    Return the Q-update for DND updates
    """
    return q_initial + self.q_learning_rate * (q_n - q_initial)

  def update(self):
    """
    Iterate through the transition queue and make NEC updates
    """
    for t in range(len(self.transition_queue)):
      transition = self.transition_queue[t]
      state = Variable(Tensor(transition.state)).unsqueeze(0)
      action = transition.action
      state_embedding = self.embedding_network(state)
      dnd = self.dnd_list[action]

      Q_N = self.Q_lookahead(t)

      if not dnd.is_present(state_embedding):
        # DND insert
        dnd.upsert(state_embedding, Q_N)
      else:
        # DND update
        Q = self.Q_update(dnd.get_value(state_embedding), Q_N)
        dnd.upsert(state_embedding, Q)

      self.replay_memory.push(state, action, Q_N)

      if t % self.update_period == 0 and not self.before_learning:
        # Train on random mini-batch from self.replay_memory
        batch = self.replay_memory.sample(self.batch_size)
        actual = torch.cat([sample.Q_N for sample in batch])
        predicted = torch.cat([self.dnd_list[sample.action].lookup(self.embedding_network(sample.state)) for sample in batch])
        loss = torch.dist(actual, predicted)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    # Clear out transition queue
    self.transition_queue = []

  def train(self):
    """
    Train an NEC agent
    For every episode, interact with environment on-policy and append all (state, action, reward) transitions to transition queue
    Call update at the end of every episode
    """
    state = self.env.reset()
    state_embedding = self.embedding_network(Variable(Tensor(state)).unsqueeze(0))
    num_episodes = 0
    for t in count():
      self.before_learning = t < self.warmup_period
      epsilon = self.epsilon_schedule(t)
      action = self.choose_action(state_embedding, epsilon)
      next_state, reward, done, info = self.env.step(action)
      self.transition_queue.append(Transition(state, action, reward))
      state = next_state
      if done:
        num_episodes += 1

        state = self.env.reset()
        self.update()

        if self.test_period is not None and num_episodes % self.test_period == 0:
          print("Evaluation avg reward: {}".format(self.test()))
          state = self.env.reset()
 
      state_embedding = self.embedding_network(Variable(Tensor(state)).unsqueeze(0))

  def test(self, nb_episodes=100, maximum_episode_length=500):
    def evaluate_episode():
      reward = 0
      observation = self.env.reset()
      for _ in range(maximum_episode_length):
        action = self.choose_action(self.embedding_network(Variable(Tensor(observation)).unsqueeze(0)), 0)
        observation, immediate_reward, finished, info = self.env.step(action)
        reward += immediate_reward
        if finished:
          break
      return reward

    r = 0
    for _ in range(nb_episodes):
      r += evaluate_episode()
    return r / nb_episodes
