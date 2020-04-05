import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations

#a brute force search algorithm
score = float("inf")
solution = []

fh = open('410_50_info.txt', "r+")
fh_solution = open('400_410_sol.txt', "w")

lines=fh.readlines()


for graph_index in range(400,410):
    print(graph_index)
    
    graph_name_str = lines[graph_index].split()[0]
    graph_info_str = str(lines[graph_index]).rstrip()
    print(graph_name_str)
    
    graph = nx.read_edgelist(graph_name_str, create_using=nx.DiGraph())
    nodes_list = list(graph.nodes)
    
    #generate all possible solutions to check
    solutions_check = []
    for i in range(len(nodes_list)):
        comb = combinations(nodes_list, i) 
        for j in list(comb): 
            solutions_check.append(j)
            
    #check whether covered for each possible solution
    is_found = False
    solution_number_nodes = 0
    num_solutions = len(solutions_check)    

    for i in range(num_solutions):
        #print(solution)
        solution = solutions_check[i]
        graph = nx.read_edgelist(graph_name_str, create_using=nx.DiGraph())
        for node in solution:
            graph.remove_node(node)
            #print(graph.number_of_edges())
            if(graph.number_of_edges() == 0):
                is_found = True 
                print(solution)
                solution_number_nodes = len(solution)
                break
        if(is_found):
            print(solution_number_nodes)
            break
    
    graph_info_str = graph_info_str + " " + str(solution_number_nodes) + "\n"
    fh_solution.write(graph_info_str)

fh_solution.close()
fh.close()
