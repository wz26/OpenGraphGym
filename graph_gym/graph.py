#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:06:44 2019

@author: weijianzheng

graph.py: This class is used to represent the graph, 
          it works as the interface between graph environment 
              and several different types of the graph object 
              (e.g., Basic_graph, networkx graph)

"""

import networkx as nx
import numpy as np
import copy
from node2vec import Node2Vec
from graph_gym.basic_graph import Basic_graph

from gensim.models import KeyedVectors

class Graph:
    
    def __init__(self, env_type, file_name, num_embed_dim, limit_nodes):
        
        self.env_type  = env_type
        self.file_name = file_name    
        # used for testing on different size graph
        self.limit_nodes = limit_nodes
        
        # Need to initialize different types of the graph 
        #   based on env_type         
        if(self.env_type == 'merge_cyclomatic'): 
            # this game just need basic graph
            self.b_graph = Basic_graph(self.file_name, False)
            
        if(self.env_type == 'merge_clustering_cv'): 
            # this game use networkx 
            self.b_graph = Basic_graph(self.file_name, False)
            
            #print(self.file_name)
            self.edgelist_file_name = self.file_name.split(".")[0]
            self.edgelist_file_name = self.edgelist_file_name + ".edgelist"
            
            #print(self.edgelist_file_name)
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, \
                                           create_using=nx.DiGraph())
            
            self.old_cv =  nx.average_clustering(self.temp_G.to_undirected())
            
            print("The original cv is " + str(self.old_cv))
            
        if(self.env_type == 'merge_combine'): 
            # this game use networkx 
            self.b_graph = Basic_graph(self.file_name, False)
            
            #print(self.file_name)
            self.edgelist_file_name = self.file_name.split(".")[0]
            self.edgelist_file_name = self.edgelist_file_name + ".edgelist"
            
            #print(self.edgelist_file_name)
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, \
                                           create_using=nx.DiGraph())
            
            self.old_cv =  nx.average_clustering(self.temp_G.to_undirected())
            
            print("The original cv is " + str(self.old_cv))
            
        if(self.env_type == 'greedy_min_cover'): 
            # this game use networkx 
            #self.b_graph = Basic_graph(self.file_name, False)
            
            edgelist_file = file_name.split("/")

            edgelist_file_name_temp = edgelist_file[len(edgelist_file)-1].\
                    split(".")[0]
            edgelist_file_name_temp += ".edgelist"

            self.edgelist_file_name = file_name.replace(edgelist_file[\
                    len(edgelist_file)-1], edgelist_file_name_temp)
            
            #print(self.edgelist_file_name)
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, \
                                           create_using=nx.DiGraph())
            
            #self.temp_G = nx.fast_gnp_random_graph(20, 0.6, seed=None, \
                    #directed=False)
            #self.ori_G = self.temp_G 
            
            # the number of nodes in graph
            self.num_nodes = len(self.temp_G)
            self.nodes_list = list(self.temp_G.nodes)
            
            # begin to prepare for the embedding
            # graph embedding parameters: 
            self.num_embed_dim = num_embed_dim
            self.num_walk_length = 5
            self.num_num_walks = 100
            self.num_workers = 4
            
            # Precompute probabilities and generate walks 
            # Use temp_folder for big graphs
            self.node2vec = Node2Vec(self.temp_G, \
                                     dimensions = self.num_embed_dim, \
                                     walk_length = self.num_walk_length, \
                                     num_walks = self.num_num_walks, \
                                     workers = self.num_workers, p=256, q=0.25, \
                                     quiet=True)  
            
            # Embed nodes
            self.model = self.node2vec.fit(window=2, min_count=1, \
                                           batch_words=4)  
            # Any keywords acceptable by gensim.Word2Vec can be passed, 
            # `dimensions` and `workers` are automatically passed 
            # (from the Node2Vec constructor)
            
            # initialize a sigma: used for later agent
            # state + candidate node           
            self.state = np.zeros(shape = (1, self.num_embed_dim*2))
            # the number of nodes in the current solution
            self.s_count = 0
             
            # the number of edges in the current solution
            self.s_edge_count = 0
            # the number of edges in the graph:
            self.num_edges = self.temp_G.number_of_edges()
            
            self.edgelist = list(self.temp_G.edges)
            
            # it is a collection of nodes in the current partial solution
            self.partial_solution = []
            
            for i in range(0, self.num_edges):
                self.edgelist[i] += (0,)

        if(self.env_type == 'min_cover_s2v'):               
 
            edgelist_file = file_name.split("/")

            edgelist_file_name_temp = edgelist_file[len(edgelist_file)-1].\
                    split(".")[0]
            edgelist_file_name_temp += ".edgelist"

            self.edgelist_file_name = file_name.replace(edgelist_file[\
                    len(edgelist_file)-1], edgelist_file_name_temp)
            
            #print(self.edgelist_file_name)
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name)          
            # one of the input for this game
            self.ad_matrix = nx.to_numpy_matrix(self.temp_G)

            # the number of nodes in graph
            self.num_nodes = len(self.temp_G)
            self.nodes_list = list(self.temp_G.nodes)
            
            self.node_selected = np.zeros(shape = (1, self.limit_nodes)) 
            self.node_measured = np.zeros(shape = (1, self.limit_nodes, 1))
            self.node_all = np.ones(shape = (1, self.limit_nodes, 1))
            
            self.ad_matrix = np.lib.pad(self.ad_matrix, ((0,self.limit_nodes-self.num_nodes), \
                                                         (0,self.limit_nodes-self.num_nodes)), \
                                        'constant', constant_values=(0))
            self.ad_matrix = np.reshape(self.ad_matrix, (-1, self.limit_nodes, self.limit_nodes))

            self.state = []
            self.state.append(self.node_selected)
            self.state.append(self.ad_matrix)
            self.state.append(self.node_measured)
            self.state.append(self.node_all)
            # the number of nodes in the current solution
            self.s_count = 0
             
            # it is a collection of nodes in the current partial solution
            self.partial_solution = []
            
        if(self.env_type == 'max_cut_s2v'):               
 
            edgelist_file = file_name.split("/")

            edgelist_file_name_temp = edgelist_file[len(edgelist_file)-1].\
                    split(".")[0]
            edgelist_file_name_temp += ".edgelist"

            self.edgelist_file_name = file_name.replace(edgelist_file[\
                    len(edgelist_file)-1], edgelist_file_name_temp)
            
            #print(self.edgelist_file_name)
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name)          
            # one of the input for this game
            self.ad_matrix = nx.to_numpy_matrix(self.temp_G)

            # the number of nodes in graph
            self.num_nodes = len(self.temp_G)
            self.nodes_list = list(self.temp_G.nodes)
            
            self.node_selected = np.zeros(shape = (1, self.limit_nodes)) 
            self.node_measured = np.zeros(shape = (1, self.limit_nodes, 1))
            self.node_all = np.ones(shape = (1, self.limit_nodes, 1))
            
            self.ad_matrix = np.lib.pad(self.ad_matrix, ((0,self.limit_nodes-self.num_nodes), \
                                                         (0,self.limit_nodes-self.num_nodes)), \
                                        'constant', constant_values=(0))
            self.ad_matrix = np.reshape(self.ad_matrix, (-1, self.limit_nodes, self.limit_nodes))

            self.state = []
            self.state.append(self.node_selected)
            self.state.append(self.ad_matrix)
            self.state.append(self.node_measured)
            self.state.append(self.node_all)
            # the number of nodes in the current solution
            self.s_count = 0
             
            # it is a collection of nodes in the current partial solution
            self.partial_solution = []      
            
    def update_state(self, action):
        
        if(self.env_type == 'merge_cyclomatic'): 
            # this game just need basic graph
            self.done = self.b_graph.get_new_state_weight(action[1], \
                                                          action[2])
            self.state = self.graph.b_graph.state
   
        if(self.env_type == 'merge_clustering_cv'): 
            # this game need both basic and networkx graph
            self.done = self.b_graph.fast_get_new_state(action[1], action[2])
            
            first_node = self.b_graph.function_list[action[1]]
            second_node = self.b_graph.function_list[action[2]]
            
            temp_GG = nx.contracted_nodes(self.temp_G, first_node, \
                                          second_node)
            self.temp_G.clear()
            self.temp_G = temp_GG.copy()
            
            self.state = self.graph.b_graph.state
            
        if(self.env_type == 'merge_combine'): 
            # this game need both basic and networkx graph
            self.done = self.b_graph.get_new_state(action[1], action[2])
            
            first_node = self.b_graph.function_list[action[1]]
            second_node = self.b_graph.function_list[action[2]]
            
            temp_GG = nx.contracted_nodes(self.temp_G, first_node, \
                                          second_node)
            self.temp_G.clear()
            self.temp_G = temp_GG.copy()
            
            self.state = self.graph.b_graph.state
            
        if(self.env_type == 'greedy_min_cover'): 
            
            # first add to solution's vector
            self.state[0][0:self.num_embed_dim] += self.model.wv[\
                     self.nodes_list[action]]
            
            self.state[0][self.num_embed_dim:self.num_embed_dim*2] = \
            self.model.wv[self.nodes_list[action]]
            
            # update the nodes in the current partial solution
            self.partial_solution.append(action)
            
            self.s_count += 1
            self.done = False
            
            if(self.s_count >= self.num_nodes): 
                self.done = True
                
            node_name = self.nodes_list[action]
            print("selected node is " + str(node_name))
            #print(self.edgelist)
            
            # Need to check the number of edges to decide continue or not
            for i in range(len(self.edgelist)):
                edge = self.edgelist[i]
                edge_checked = len(edge)
                #print(len(edge))
                #print("two nodes are " + str(edge[0]) + " and " + \
                #      str(edge[1]) + "\n")
                if((edge[0] == node_name or edge[1] == node_name) and \
                   edge_checked == 3):
                    self.s_edge_count += 1
                    #print(edge)
                    self.edgelist[i] += (1,) 
                    
            if(self.s_edge_count >= self.num_edges):
                self.done = True
                
            #print("there are " + str(self.s_edge_count) + " edges now")
            #self.reward = self.temp_G.degree[self.nodes_list[action]]
            
            # update the graph
            self.temp_G.remove_node(self.nodes_list[action])
            
            # After adding to partial solution, remove it from the 
            # candiate nodes and update the embedding
            # Precompute probabilities and generate walks 
            # Use temp_folder for big graphs
            self.node2vec = Node2Vec(self.temp_G, \
                                     dimensions = self.num_embed_dim, \
                                     walk_length = self.num_walk_length, \
                                     num_walks = self.num_num_walks, \
                                     workers = self.num_workers, p=256, q=0.25, \
                                     quiet=True)  
            
            # Embed nodes
            self.model = self.node2vec.fit(window=2, min_count=1, \
                                           batch_words=4)  
             
        if(self.env_type == 'min_cover_s2v'): 
            # update the nodes in the current partial solution
            self.partial_solution.append(action)

            self.s_count += 1
            self.done = False

            if(self.s_count >= self.num_nodes):
                self.done = True

            # update the graph
            self.temp_G.remove_node(self.nodes_list[action])
            
            node_name = self.nodes_list[action]
            
            # update adjacency matrix
            self.ad_matrix[0,:,action] = 0
            self.ad_matrix[0,action,:] = 0
           
            self.state[1] = self.ad_matrix
            self.node_selected[0][action] = 1
            self.node_all[0][action] = 0
            
            self.state[0][0][action] = 1
            self.state[2] = np.zeros(shape = (1, self.limit_nodes, 1))
            #update the all vector
            self.state[3][0][action] = 0

            # check the number of edges to decide continue or not
            if(self.temp_G.number_of_edges() == 0):
                self.done = True
                
        if(self.env_type == 'max_cut_s2v'): 
            # update the nodes in the current partial solution
            self.partial_solution.append(action)

            self.s_count += 1
            self.done = False

            if(self.s_count >= self.num_nodes):
                self.done = True

            # update the graph
            self.temp_G.remove_node(self.nodes_list[action])
            
            node_name = self.nodes_list[action]
            
            # update adjacency matrix
            self.ad_matrix[0,:,action] = 0
            self.ad_matrix[0,action,:] = 0
           
            self.state[1] = self.ad_matrix
            self.state[0][0][action] = 1
            self.state[2] = np.zeros(shape = (1, self.limit_nodes, 1))

            # check the number of edges to decide continue or not
            if(self.temp_G.number_of_edges() == 0):
                self.done = True

        return self.done
        #return False

    def get_reward(self):
        
        if(self.env_type == 'merge_cyclomatic'): 
            # this game just need basic graph
            self.reward = self.b_graph.get_complexity_reward()
            self.reward *= 10
        
        if(self.env_type == 'merge_clustering_cv'): 
            
            self.new_cv = nx.average_clustering(self.temp_G.to_undirected())
            
            self.reward = self.new_cv - self.old_cv
            self.reward *= 5
            
            self.old_cv = self.new_cv
        
        if(self.env_type == 'merge_combine'): 
            
            self.new_cv = nx.average_clustering(self.temp_G.to_undirected())
            
            self.reward = self.new_cv - self.old_cv
            self.reward *= 5
            
            self.old_cv = self.new_cv
            
            temp_reward = self.b_graph.get_complexity_reward()
            temp_reward *= 10
            
            self.reward += temp_reward
            
        if(self.env_type == 'greedy_min_cover'): 
            
            #TODO: count #edges to decide the reward
            #self.reward = -10 * self.s_count
            #self.reward = -10  * (self.s_count * self.s_count + self.s_count) / 2
            #self.reward = 10 * (self.num_nodes - self.s_count)
            self.reward = 0
            #self.reward = self.reward
            #self.reward -= self.s_count
            #self.reward = -10
            #print(self.reward)
            if(self.done):
               #self.reward += 100             # give bonus for done
               self.reward -= self.s_count # give penalty for more nodes
         
        if(self.env_type == 'min_cover_s2v'):    
            #TODO: count #edges to decide the reward
            #self.reward = -10 * self.s_count
            #self.reward = -10  * (self.s_count * self.s_count + self.s_count) / 2
            #self.reward = 10 * (self.num_nodes - self.s_count)
            self.reward = 0
            #self.reward = self.reward
            #self.reward -= self.s_count
            #self.reward = -10
            #print(self.reward)
            
        return self.reward
            
            
    def reload(self):
        
        if(self.env_type == 'merge_cyclomatic'):
            self.b_graph = Basic_graph(self.file_name, True)
            
        if(self.env_type == 'merge_clustering_cv'):
            self.b_graph = Basic_graph(self.file_name, True) 
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, \
                                           create_using=nx.DiGraph())    
            self.old_cv =  nx.average_clustering(self.temp_G.to_undirected())
            
        if(self.env_type == 'merge_combine'):
            self.b_graph = Basic_graph(self.file_name, True) 
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, \
                                           create_using=nx.DiGraph())    
            self.old_cv =  nx.average_clustering(self.temp_G.to_undirected())
        
        if(self.env_type == 'greedy_min_cover'): 
            
            # reload graph
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, \
                                           create_using=nx.DiGraph())
            
            #self.temp_G = self.ori_G
            
            # the number of nodes in graph
            self.num_nodes = len(self.temp_G)
            self.nodes_list = list(self.temp_G.nodes)
            
            # initialize a sigma: used for later agent
            # state + candidate node           
            self.state = np.zeros(shape = (1, self.num_embed_dim*2))
            # the number of nodes in the current solution
            self.s_count = 0
            
            # the number of edges in the current solution
            self.s_edge_count = 0
            # the number of edges in the graph:
            self.num_edges = self.temp_G.number_of_edges()
            
            self.edgelist = list(self.temp_G.edges)
            
            self.partial_solution = []
            
            for i in range(0, self.num_edges):
                self.edgelist[i] += (0,) 
                
            self.node2vec = Node2Vec(self.temp_G, \
                                     dimensions = self.num_embed_dim, \
                                     walk_length = self.num_walk_length, \
                                     num_walks = self.num_num_walks, \
                                     workers = self.num_workers, p=256, q=0.25, \
                                     quiet=True)  
            
            # Embed nodes
            self.model = self.node2vec.fit(window=2, min_count=1, \
                                           batch_words=4)  
 
        if(self.env_type == 'min_cover_s2v'):               
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, )
            
            # one of the input for this game
            self.ad_matrix = nx.to_numpy_matrix(self.temp_G)

            #self.temp_G = nx.fast_gnp_random_graph(20, 0.6, seed=None, \
                    #directed=False)
            #self.ori_G = self.temp_G 
            
            # the number of nodes in graph
            self.num_nodes = len(self.temp_G)
            self.nodes_list = list(self.temp_G.nodes)

            self.node_selected = np.zeros(shape = (1, self.limit_nodes)) 
            self.node_measured = np.zeros(shape = (1, self.limit_nodes, 1))
            self.node_all = np.ones(shape = (1, self.limit_nodes, 1))
            
            self.ad_matrix = np.lib.pad(self.ad_matrix, ((0,self.limit_nodes-self.num_nodes), \
                                                         (0,self.limit_nodes-self.num_nodes)), \
                                        'constant', constant_values=(0))
            self.ad_matrix = np.reshape(self.ad_matrix, (-1, self.limit_nodes, self.limit_nodes))
        
            self.state = []
            self.state.append(self.node_selected)
            self.state.append(self.ad_matrix)
            self.state.append(self.node_measured)
            self.state.append(self.node_all)
            # the number of nodes in the current solution
            self.s_count = 0
             
            # it is a collection of nodes in the current partial solution
            self.partial_solution = []
     
                          
    # embed a node
    def embed_node(self, node_index):  
        if(self.env_type == 'min_cover_s2v'):
            temp_states = [] 
            temp_states.append(np.zeros(shape = (1, self.limit_nodes)))
            temp_states.append(np.zeros(shape = (1, self.limit_nodes, self.limit_nodes)))
            temp_states.append(np.zeros(shape = (1, self.limit_nodes, 1)))
            temp_states.append(np.ones(shape = (1, self.limit_nodes, 1)))
            
            temp_states[0][0] = copy.deepcopy(self.state[0])
            temp_states[1][0] = copy.deepcopy(self.ad_matrix)
            temp_states[3][0] = copy.deepcopy(self.state[3])
            temp_states[2][0][node_index][0] = 1
            
            return temp_states
        else:
            embed_vec = np.zeros(shape = (1, self.num_embed_dim*2))
            embed_vec[0][0:self.num_embed_dim] = self.state[0][0:self.num_embed_dim]
            
       
            embed_vec[0][self.num_embed_dim:2*self.num_embed_dim] = self.model.wv[\
                     self.nodes_list[node_index]]
        
            return embed_vec
           
    # embed multiple nodes
    def embed_nodes_multiple(self, nodes_index):  
        if(self.env_type == 'min_cover_s2v'):

            batch_size = len(nodes_index)
            #print(num_testing)
            node_index = 0
            
            temp_states = [] 
            temp_states.append(np.zeros(shape = (batch_size, self.limit_nodes)))
            temp_states.append(np.zeros(shape = (batch_size, self.limit_nodes, self.limit_nodes)))
            temp_states.append(np.zeros(shape = (batch_size, self.limit_nodes, 1)))
            temp_states.append(np.ones(shape = (batch_size, self.limit_nodes, 1)))
            
            for candidate_node in nodes_index: 
                temp_states[0][node_index] = copy.deepcopy(self.state[0])
                temp_states[1][node_index] = copy.deepcopy(self.ad_matrix)
                temp_states[2][node_index][candidate_node][0] = 1
                temp_states[3][node_index] = copy.deepcopy(self.state[3])
                node_index += 1 
                
            return temp_states
        else:
            embed_vec = np.zeros(shape = (1, self.num_embed_dim*2))
            embed_vec[0][0:self.num_embed_dim] = self.state[0][0:self.num_embed_dim]
            
       
            embed_vec[0][self.num_embed_dim:2*self.num_embed_dim] = self.model.wv[\
                     self.nodes_list[node_index]]
        
            return embed_vec
                        
