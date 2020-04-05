#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 04 2019

graph_generator.py: 

This script is used to generate multiple graphs.

Graph type, #graphs, edge possibility can be specified
Output: multiple graphs: with an edgelist and txt file
        a information file: graph_name, #nodes, #edges, 
            2-opt solution, greedy solution, brute force solution  
            
How to run: py graph_generator #graphs #nodes #edge_possibility #type 
        
@author: weijianzheng
"""

import sys
import networkx as nx
import numpy as np
from networkx.algorithms.approximation import min_weighted_vertex_cover

print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))

num_graphs = int(sys.argv[1])
num_nodes = int(sys.argv[2])
edge_possibility = float(sys.argv[3])
graph_type = sys.argv[4]

print("number of graphs is " + str(num_graphs) + "\n")
print("number of edges is " + str(num_nodes) + "\n")
print("edge possibility is " + str(edge_possibility) + "\n")
print("graph type is " + str(graph_type) + "\n")

info_file_name = str(num_graphs) + "_"
info_file_name += str(num_nodes)
info_file_name += "_info.txt"

fh = open(info_file_name, "w")

for i in range(num_graphs):
    print("generating " + str(i) + "th graph ...")
    
    if(graph_type == "er"):
        graph = nx.erdos_renyi_graph(num_nodes, edge_possibility)
        #graph = nx.gnp_random_graph(num_nodes, edge_possibility)
    
    num_edges = len(graph.edges)
    print("there are " + str(num_edges) + " edges")
    
    cover = min_weighted_vertex_cover(graph)
    two_opt = len(cover)
    
    file_name = graph_type + str(num_nodes)
    #TODO: change the 015 here:
    file_name = file_name + "_015_" + str(num_edges) + "_"
    file_name = file_name + str(two_opt) + "_v" + str(i)
    
    file_name_temp = file_name + ".edgelist"
    file_name = file_name + ".txt"
    
    nx.write_edgelist(graph, file_name)
    nx.write_edgelist(graph, file_name_temp)
    
    goal = graph.number_of_edges()
    num_nodes = len(graph)
    nodes_list = list(graph.nodes)

    num_steps = 0

    for step in range(num_nodes):
        candidate_nodes = np.arange(len(graph))
        score = float("-inf")
        # a greedy algorithm solution: 
        for candidate_node in candidate_nodes:
            #print(candidate_node)
            new_score = graph.degree[nodes_list[candidate_node]]
            #print(new_score)
            if(new_score >= score):
                score = new_score
                action = candidate_node

        #print(graph.degree[nodes_list[action]])
        graph.remove_node(nodes_list[candidate_node])
        num_steps += 1
        state = graph.number_of_edges()
        if(state == 0):
            #print(action)
            break

    greedy_sol = num_steps
    
    #nx.write_edgelist(graph, "ER_50_nodes/er50_015_169_48_v0.edgelist")
    #nx.write_edgelist(graph, "ER_50_nodes/er50_015_169_48_v0.txt")
    
    info_temp = file_name + " " + str(num_nodes) + " " + str(num_edges) + " "
    info_temp = info_temp + str(two_opt) + " " + str(greedy_sol)   
    fh.write(info_temp  + " \n")
    
    print("done with generating " + str(i) + "th graph.")
    
fh.close()
    
    