#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:56:38 2019

@author: weijianzheng
"""

import numpy as np

class Greedy_graph:
    
    def __init__(self, file_name, reset):
        
            
        if(reset == False):
            print("Game info: ")
            print("The size of state matrix is [" + str(h) + "," + str(w) + "]")
            print("There are " + str(h) + " functions")
            print("The original complexity is " + str(self.old_complexity))
        
    # update the state vector
    def get_new_state(self, first_node_index, second_node_index):
          

        
    # update the state vector
    def fast_get_new_state(self, first_node_index, second_node_index):
          
        temp_index = 0

        if(first_node_index > second_node_index):
            temp_index = first_node_index
            first_node_index = second_node_index 
            second_node_index = temp_index
    
        for i in range(len(self.function_list)):
            # mark the second merge node's new module index (-1)
            if(i == second_node_index):
                self.state[i][len(self.function_list)] = -1
                   
        done = True
        node_count = 0
        
        for i in range(len(self.function_list)):
            if(self.state[i][len(self.function_list)] != -1):
                node_count += 1
                
        if(node_count >= 1):
            done = False
                    
        return done
                    
        print(self.state)
        
    # update the state with weight 
    def get_new_state_weight(self, first_node_index, second_node_index):  
        
        temp_index = 0

        if(first_node_index > second_node_index):
            temp_index = first_node_index
            first_node_index = second_node_index 
            second_node_index = temp_index
    
        for i in range(len(self.function_list)):
            # mark the second merge node's new module index (-1)
            if(i == second_node_index):
                self.state[i][len(self.function_list)] = -1
        
            for j in range(len(self.function_list)):
                # first remove the inside calling
                if((i == first_node_index and j == second_node_index) or \
                   (i == second_node_index and j == first_node_index)):
                    self.state[i][j] = 0
                    continue
            
                # then consider second node's calling
                if(i == second_node_index and self.state[i][j] >= 1):
                    self.state[i][j] = 0
                    self.state[first_node_index][j] += 1
            
                # then consider second node's calling
                if(j == second_node_index and self.state[i][j] >= 1):
                    self.state[i][j] = 0
                    self.state[i][first_node_index] += 1
                    
        #print(state)
        
        done = False
        node_count = 0
        
        for i in range(len(self.function_list)):
            if(self.state[i][len(self.function_list)] != -1):
                node_count += 1
                
        if(node_count >= 1):
            done = True
                    
        return done 
    
    # return the reward
    def get_complexity_reward(self):
        
        # check the system complexity:
        system_complexity = 0
        
        #print(self.state)

        for i in range(len(self.module_list)):
            module_index = i
            complexity_edges = 0
            complexity_nodes = 0
            
            # go through each row of the matrix
            for j in range(len(self.function_list)):
                first_module_index = self.state[j][len(self.function_list)]
                if(first_module_index == module_index):
                    complexity_nodes += 1
                    # go through each column of the matrix
                for k in range(len(self.function_list)):
                    second_module_index = self.state[k][len(\
                                                    self.function_list)]
                    if(self.state[j][k] < 1):
                        continue
                
                    if(first_module_index != second_module_index and \
                       (first_module_index == module_index or \
                        second_module_index == module_index)):
                        complexity_edges += 1
                          
            module_complexity = complexity_edges - complexity_nodes + 2
            #print("module " + str(i) + " has " + str(complexity_nodes) + \
            #      " nodes and " + str(complexity_edges) + " edges")
            if(complexity_nodes == 0):
                #print("module " + str(i) + " is skipped because of no node")
                continue
        
            #print(module_complexity)
            system_complexity += module_complexity
    
        #print("The old system complexity is: " + str(self.old_complexity))
        #print("The new system complexity is: " + str(system_complexity)) 
    
        reward = self.old_complexity - system_complexity
        self.old_complexity = system_complexity
        
        #reward *= 0.1
    
        return reward