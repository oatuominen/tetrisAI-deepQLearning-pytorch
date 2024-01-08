import torch as T
import pygame
import json
import copy
import operator
import numpy as np
import random as r
from collections import deque
import queue
#from game import Game
from deepqnetwork import DeepQNetwork



class Agent:

    def __init__(self, learning_rate, discount_factor, epsilon):
        self.learning_rate = learning_rate      # step size, adjusting magnitude of improvement
        self.discount_factor = discount_factor  # determines the value of immediate reward in contrast to future rewards
        self.epsilon = epsilon                  # exploitation-exploration -rate

        self.mem_size = 50000  
        self.batch_size = 512    
        self.epsilon_decay = 0.999 
        self.min_epsilon = 0.001
        #self.epsilon_end_episode = 2000
        #self.epsilon_decay = (self.epsilon - self.min_epsilon) / self.epsilon_end_episode

        # Deep Q Network:        
        self.dqn = DeepQNetwork(learning_rate=self.learning_rate)
        self.dqn.load_dqn()

        self.replay_buffer = deque(maxlen=self.mem_size)



    def store_transition(self, initial_state, action, next_state, reward, done):
        self.replay_buffer.append([initial_state, action, next_state, reward, done])


    def random_journey(self): return np.random.rand() < self.epsilon


    def learn(self):
        #if len(self.replay_buffer) < self.batch_size: return
        batch_size = min(self.batch_size, len(self.replay_buffer))

        batch = r.sample(self.replay_buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        state_batch = T.tensor(states, dtype=T.float32).to(self.dqn.device)            
        next_state_batch = T.tensor(next_states, dtype=T.float32).to(self.dqn.device)
        reward_batch = T.tensor(rewards)[:, None].to(self.dqn.device) 
        done_batch = T.tensor(dones).to(self.dqn.device) 
        
        q_values = [self.dqn.forward(state) for state in state_batch]
        self.dqn.eval()
        q_next = [self.dqn.forward(state) for state in next_state_batch]

        self.dqn.train()

        q_target = T.cat(
            tuple((reward if done else reward + self.discount_factor * prediction.item()) 
                  for reward, done, prediction in zip(reward_batch, done_batch, q_next))
        )[:, None]


        q_values = [[value.item()] for value in q_values]
        q_target = [[value.item()] for value in q_target]

        q_values = T.tensor(q_values, requires_grad=True)
        q_target = T.tensor(q_target, requires_grad=True)

        self.dqn.optimizer.zero_grad()
        loss = self.dqn.loss(q_values, q_target).to(self.dqn.device)
        loss.backward()
        self.dqn.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)



        



        







