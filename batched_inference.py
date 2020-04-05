#!/usr/bin/env python
# coding: utf-8

import sys

sys.executable

# sys.path.remove('/usr/bin/python3')
# sys.path.append('/home/wwz/anaconda3/envs/horovod_env/bin/python')

# sys.executable

#multiple envs test of 20-to-20
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
import time
import horovod.tensorflow as hvd

from collections import deque
from operator import itemgetter

from graph_gym.agent import Graph_agent
from graph_gym.graph_env import Graph_env

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

hvd.init()

print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))

batch_size = int(sys.argv[1])
train_limit = int(sys.argv[2])
test_limit = int(sys.argv[3])

size_str = str(train_limit) + "_" + str(test_limit)
fh_r = open("er" + size_str + "_epsilondecay_e64_test_reward_p1" + str(batch_size) + ".txt", "w")

fh0 = open("er_graph_" + str(train_limit) + "_410/410_" + str(train_limit) + "_info.txt", "r")
fh1 = open("er_graph_" + str(test_limit) + "_410/410_" + str(test_limit) + "_info.txt", "r")

lines0 = fh0.readlines()
lines1 = fh1.readlines()

# now need to create a greedy environment 
num_embed_dim = 64

total_episodes = 1500          # Total episodes
test_steps = 500              # Max steps per episode

learning_rate = 0.000001      # Learning rate
gamma = 0.9                   # Discounting rate

# Exploration parameters
epsilon = 0.1
max_epsilon = 0.9             # Exploration probability at start
min_epsilon = 0.1            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

game_info = ['min_cover_s2v', train_limit, num_embed_dim, 1, epsilon, learning_rate, gamma]
agent = Graph_agent(game_info)
    
rewards = []
old_reward = 0

replay_buffer = deque(maxlen=50000)

rewards_window = deque(maxlen=20)
rewards_windows = []

t = time.time()
testing_time = 0
backward_time = 0
time_tmp = 0

batch_size = int(batch_size/hvd.size())

# 2 For life or until learning is stopped
for episode in range(total_episodes):

    if(hvd.rank() == 0):
        print("now start episode "+str(episode))

    candidate_nodes = []
    train_candidate_nodes = []

    random.seed(episode+hvd.rank()) 

    graph_index = int(episode)%400
    graph_name = "er_graph_" + str(train_limit) + "_410/" + lines0[graph_index].split()[0]
    
    print(graph_name)
    env = Graph_env('min_cover_s2v', graph_name, num_embed_dim, train_limit)

    rewards_buffer = deque(maxlen=test_steps)
    states_buffer = deque(maxlen=test_steps)

    # print("done with getting original reward")
    done = False
    test_total_rewards = 0

    env.reset()
    state = env.graph.state
    old_state = state

    # a list of candidate nodes
    candidate_nodes = np.arange(env.graph.num_nodes)
    train_candidate_nodes = np.arange(env.graph.num_nodes)

    temp_replay_buffer = []

    for step in range(test_steps):
        
        #print("prepare to select node")
        score = float("-inf")
        action = 0
        
        random.seed(step+hvd.rank()) 
        exp_tradeoff = random.uniform(0,1)
        
        count = 0
        
        #print(candidate_nodes)
        if(exp_tradeoff > epsilon):
            temp_states = env.embed_nodes_multiple(candidate_nodes)
            new_scores = agent.greedy_act_multiple(temp_states, step)
           
            action_index = np.argmax(new_scores) 
            action = candidate_nodes[action_index]
            score = new_scores[action_index]
            state = env.embed_node(action) 
        else:
            action = np.squeeze(np.random.choice(candidate_nodes, 1))
            temp_state = env.embed_node(action)
            score = agent.greedy_act(temp_state, step)
            state = temp_state.copy()

        # remove selected node from the candidate nodes
        action_index = np.argwhere(candidate_nodes == action)
        train_action_index = np.argwhere(train_candidate_nodes == action)
        
        candidate_nodes = np.delete(candidate_nodes, action_index)
        
        train_candidate_nodes = np.delete(train_candidate_nodes, train_action_index)
        
        next_state, reward, done = env.step(action)

        rewards_buffer.append(copy.deepcopy(reward))
        states_buffer.append(copy.deepcopy(state))

        test_total_rewards += reward

        if(step >= 4):
            #print("it is done")
            if(hvd.rank() == 0 and done):
                print("it takes " + str(step+1) + " steps")
#                 fh_r1.write(str(step+1)  + " \n")
            if done:           
                state = next_state
                for exp in temp_replay_buffer:
                    exp[1] -= 1
                    exp.pop(2)
                    replay_buffer.append(copy.deepcopy(exp))
                    temp_replay_buffer.remove(exp)
                break
            #here we need to update the previous label
            for exp in temp_replay_buffer:
                exp[1] -= 1
                exp[2] -= 1
                if(exp[2] <= -5):
                    exp.pop(2)
                    replay_buffer.append(copy.deepcopy(exp))
                    temp_replay_buffer.remove(exp)

        score = float("-inf")
        action = 0
        
        temp_states = env.embed_nodes_multiple(train_candidate_nodes)
        new_scores = agent.greedy_act_multiple(temp_states, step)

        action_index = np.argmax(new_scores) 
        action = train_candidate_nodes[action_index]
        score = new_scores[action_index]

        label = reward       
        label += gamma * score

        experience = []

        experience.append(copy.deepcopy(state))
        experience.append(copy.deepcopy(label))
        experience.append(0)

        temp_replay_buffer.append(copy.deepcopy(experience))

        old_state = state
        old_reward = reward
        state = next_state

        if(len(replay_buffer) >= batch_size):
            random.seed(episode%10) 
            #exp_index = random.sample(range(0, len(replay_buffer)), num_gpus)
            exp_index = random.sample(range(0, len(replay_buffer)), batch_size)

            #t1 = time.time()
            sampled_exp = itemgetter(*exp_index)(replay_buffer)

            t1 = time.time()
            agent.nn_model.backward_batch(sampled_exp, batch_size)            
            #agent.nn_model.backward_batch(sampled_exp, num_gpus)
            time_tmp = time.time() - t1
            print(str(time_tmp))
            backward_time += time_tmp

       
    # Reduce epsilon (because we need less and less exploration)
    #epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    #print(epsilon)

    if(hvd.rank() == 0):
        rewards.append(step+1) 

    if(hvd.rank() == 0 and ((episode+1)%20 == 0 or episode == 0)):
        
        t1 = time.time()
        
        test_total_rewards = deque(maxlen=10)
        done = False
        
        for test_episode in range(0, 10):
            solution = []
            
            graph_index = int(test_episode)
            graph_name = "er_graph_" + str(test_limit) + "_410/" + lines1[graph_index+400].split()[0]

            test_env = Graph_env('min_cover_s2v', graph_name, num_embed_dim, test_limit)
            score = float("-inf")
            test_action = 0 
            test_candidate_nodes = []

            test_candidate_nodes = np.arange(test_env.graph.num_nodes)

            for test_step in range(test_steps):

                test_score = float("-inf")
                test_action = 0 

                count = 0
                
                temp_states = test_env.embed_nodes_multiple(test_candidate_nodes)
                test_new_scores = agent.greedy_act_multiple(temp_states, test_step)

                test_action_index = np.argmax(test_new_scores) 

                test_action = test_candidate_nodes[test_action_index]
                solution.append(test_action)
                test_score = test_new_scores[test_action_index]

                test_action_index = np.argwhere(test_candidate_nodes == test_action)
                test_candidate_nodes = np.delete(test_candidate_nodes, test_action_index)

                next_state, reward, done = test_env.step(int(test_action))

                if done:
                    print("it takes " + str(test_step) + " steps")
                    #print(solution)
                    test_total_rewards.append(int(len(solution))) 
                    break
                    
        testing_time += time.time() - t1

        print("average is: " + str(sum(test_total_rewards)/10) + " at " + str(episode))
        fh_r.write(str(sum(test_total_rewards)/10)  + " \n")

        rewards_window.append(sum(test_total_rewards)/10)
        if(episode % 20 == 0):
            rewards_windows.append(np.mean(rewards_window))

fh_r.close()

elapsed_time = time.time() - t

print("Training time is " +  str(elapsed_time) + "for p " + str(hvd.rank()))
print("Backward time is " +  str(backward_time) + "for p " + str(hvd.rank()))
print("Testing time is " +  str(testing_time) + "for p " + str(hvd.rank()))
