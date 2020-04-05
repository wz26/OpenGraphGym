#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:38:27 2019

agent.py: graph agent class

How to use this class: 
   
    
@author: weijianzheng
"""

import numpy as np
import random

from collections import deque
from graph_gym.model import Model
from graph_gym.agent_helper import * 

class Graph_agent:
    
    def __init__(self, game_info):
        
        # create a new model
        self.env_type = game_info[0]
        
        self.model_info = []
        self.model_info = game_info[0:6]
        self.nn_model = Model(self.model_info)
        
        self.eps   = game_info[4]
        self.lr    = game_info[5]
        self.gamma = game_info[6]
        
        self.replay_buffer = deque(maxlen=5)
        
        self.candidate_nodes = []
        
    # interact with the agent_helper function, decide action based on eps
    def act(self, state, step):
        
        self.nn_model.forward(state)
        
        self.label = self.nn_model.A2.copy()
        self.label = np.asarray(self.label)
        self.label = self.label.reshape((self.model_info[3], 1)) 
        
        #print(self.nn_model.A2)
        
        ## If this number > greater than epsilon --> 
        ##   exploitation (taking the biggest Q value for this state)
        exp_tradeoff = random.uniform(0,1)
        if exp_tradeoff > self.eps:
            (first_node_index, \
             second_node_index) = find_two_nodes(state, self.nn_model.A2)
        else:
            (first_node_index, \
             second_node_index) = find_two_nodes_random(state)
       
        action = []
        action.append('')
        
        if(self.env_type == 'merge_cyclomatic' or \
           self.env_type == 'merge_clustering_cv' or \
           self.env_type == 'merge_combine'):
            action[0] = 'merge'
        
        temp = 0
        if(first_node_index > second_node_index):
           temp = first_node_index
           first_node_index = second_node_index 
           second_node_index = temp
        
        action.append(first_node_index)
        action.append(second_node_index)
        
        return action
    
    # interact with the agent_helper function, decide action based on eps
    def greedy_act(self, state, step):
               
        self.nn_model.forward(state)
        
        if(self.env_type == 'min_cover_s2v'):
            self.label = self.nn_model.A2.copy()
        else:
            self.label = self.nn_model.A2.copy()
       
        self.label = np.asarray(self.label)
       
        self.label = self.label.reshape((self.model_info[3], 1)) 

        score = np.squeeze(self.label)

        return score
    
    # interact with the agent_helper function, decide action based on eps
    def greedy_act_multiple(self, state, step):
              
        self.nn_model.forward_batch(state)
        
        if(self.env_type == 'min_cover_s2v'):
            self.label = self.nn_model.A2.copy()
        else:
            self.label = self.nn_model.A2.copy()
       
        self.label = np.asarray(self.label)

        score = np.squeeze(self.label)

        return score
    
    # interact with the agent_helper function, decide action based on model
    def test(self, state, step):
        
        self.nn_model.forward(state)
        
        self.label = self.nn_model.A2.copy()
        self.label = np.asarray(self.label)
        self.label = self.label.reshape((self.model_info[3], 1)) 
        
        (first_node_index,second_node_index) = find_two_nodes(\
                state, self.nn_model.A2)
        
        action = []
        action.append('')
        
        if(self.env_type == 'merge_cyclomatic' or \
           self.env_type == 'merge_clustering_cv' or \
           self.env_type == 'merge_combine'):
            action[0] = 'merge'
        
        temp = 0
        if(first_node_index > second_node_index):
           temp = first_node_index
           first_node_index = second_node_index 
           second_node_index = temp
        
        action.append(first_node_index)
        action.append(second_node_index)
        
        return action
        
    # save it to replay buffer
    def step(self, state, action, reward, next_state, done):
        #save this state, action, reward, next_state to replay buffer
        experience = []
        
        experience.append(state)
        experience.append(action)
        experience.append(reward)
        experience.append(next_state)
        experience.append(done)
        
        self.replay_buffer.append(experience)
        
        exp_index = random.sample(range(0, len(self.replay_buffer)), 1)
        
        return self.replay_buffer[exp_index[0]]
        
    # forward and backward function,
    def train(self, exp):
    #def train(self, state, reward, action):
        state = exp[3]
        reward = exp[2]
        action = exp[1]
        
        self.nn_model.forward(state)
        
        Y_temp = np.asarray(self.nn_model.A2)
        Y_temp = Y_temp.reshape((self.model_info[3], 1)) 
        
        (first_index_temp, second_index_temp\
         ) = find_two_nodes(state, self.nn_model.A2)
                 
        first_target = reward + self.gamma * Y_temp[first_index_temp]
        second_target = reward + self.gamma * Y_temp[second_index_temp]
        
        #print(self.label)
            
        self.label[action[1]] = first_target
        self.label[action[2]] = second_target  
        
        self.nn_model.backward(state, self.label)
        
    # forward and backward function,
    def batch_train(self, exps, batch_size):
        print(exps)
        print(batch_size)
        
        #TODO: code to convert exps to state
        
        self.nn_model.backward_batch(state, label, batch_size)
        
    # save the model
    def save_model(self):
        self.nn_model.save()
        return True
        
    # load the model
    def load_model(self):
        # load the model for the test
        self.nn_model.load()
        return True
    
    # close the model
    def close(self):
        # load the model for the test 
        self.nn_model.sess.close()
        return True
