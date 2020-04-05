#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:01:47 2019

basic_graph.py: This is our own newly defined graph type. 
                It is used by graph.py

@author: weijianzheng
"""

import numpy as np

class Basic_graph:
    
    def __init__(self, file_name, reset):
        
        self.file_name = file_name
        self.fh = open(self.file_name, "r")
        self.num_lis = sum(1 for line in open(self.file_name))
        
        # used to store all functions and edges, no duplicate
        self.function_list = []
        self.edge_list = []

        # used to check how many functions
        self.function_count = 0
        # used to check how many callers for one function
        self.caller_count = 0
        # used to check how many edges
        self.edge_count = 0

        # loop1: list all functions, edges (without duplicate) 
        for i in range (0, self.num_lis):
            temp_stn = self.fh.readline()
            temp_stn = temp_stn.strip()
    
            list_temp = temp_stn.split()
            #print(list_temp[0])
            if(list_temp[0] == 'function:'):
                # now find the function: start to process:
                temp_function = list_temp[1]
                function_temp = temp_function.split("::")
            else:
                # this means the module name is specified 
                if(len(function_temp) > 1):
                    function = function_temp[1]
                    if(function not in self.function_list):
                        self.function_list.append(function_temp[1])
                        self.function_count += 1
                # this means the module name is not specified
                else:
                    function = function_temp[0]
                    if(function not in self.function_list):
                        self.function_list.append(function_temp[0])
                        self.function_count += 1
        
                # now begin to check the caller function,
                # if it is not in function_list, add it
                call_function = list_temp[1]
                call_temp = call_function.split("::")
                #print(call_temp)
                # if there are more than callers for one module
                if(len(call_temp) > 1):
                    for i in range(1, len(call_temp)):
                        self.caller_count += 1
                        c_function = call_temp[i]
                        if(c_function not in self.function_list):
                            self.function_list.append(c_function)
                            self.function_count += 1
                        edge = function + " -> " + call_temp[i] 
                        if(edge not in self.edge_list):
                            self.edge_list.append(edge)
                            self.edge_count += 1
                # else: there is only one caller
                else:
                    self.caller_count += 1
                    c_function = call_temp[0]
                    if(c_function not in self.function_list):
                        self.function_list.append(c_function)
                        self.function_count += 1
                    edge = function + " -> " + call_temp[0]
                    if(edge not in self.edge_list):
                        self.edge_list.append(edge)
                        self.edge_count += 1
                        
        #print(self.edge_count)

        self.fh.close()
        self.fh = open(self.file_name, "r")

        # now enter the second loop, denote the modules of all functions
        self.number_functions = len(self.function_list)

        # module list, used to mark the corresponding module of the function   
        self.corresponding_module_list = ["-"]*self.number_functions 

        # module list, used to measure the number of modules
        self.module_list = []

        # used for check correctness
        have_marked = 0

        # loop2: mark the modules for all functions
        for i in range (0, self.num_lis):
            temp_stn = self.fh.readline()
            temp_stn = temp_stn.strip()
    
            list_temp = temp_stn.split()
            #print(list_temp[0])
            if(list_temp[0] == 'function:'):
                # now find the function: start to process:
                temp_function = list_temp[1]
                function_temp = temp_function.split("::")
            else:
                # this means the module name is specified 
                if(len(function_temp) > 1):
                    function = function_temp[1]
                    function_index = self.function_list.index(function)
                    if(self.corresponding_module_list[function_index] is "-"):
                        self.corresponding_module_list[function_index] = \
                        function_temp[0]
                        have_marked += 1
                    if(function_temp[0] not in self.module_list):
                        self.module_list.append(function_temp[0])
                # this means the module name is not specified
                else:
                    function = function_temp[0]
                    if(function not in self.function_list):
                        self.function_list.append(function_temp[0])
                        self.function_count += 1
        
                # now begin to check the caller function,
                # if it is not in function_list, add it
                call_function = list_temp[1]
                call_temp = call_function.split("::")
                #print(call_temp)
                # if there are more than callers for one module
                if(len(call_temp) > 1):
                    for i in range(1, len(call_temp)):
                        self.caller_count += 1
                        c_function = call_temp[i]
                        function_index = self.function_list.index(c_function)
                        if(self.corresponding_module_list[function_index] \
                           is "-"):
                            self.corresponding_module_list[function_index] \
                            = call_temp[0]
                            have_marked += 1
                        if(call_temp[0] not in self.module_list):
                            self.module_list.append(call_temp[0])

        self.fh.close()

        # now we begin to build a adjacency matrix

        # Creates a list containing #functions lists, 
        # each of #functions+1 items, all set to 0
        w, h = len(self.function_list)+1, len(self.function_list);
        self.state = np.zeros(shape=(h,w))
    
        #print(ad_matrix)

        for function in self.function_list:
            function_index = self.function_list.index(function)
            module_index = self.\
            module_list.index(self.corresponding_module_list[function_index])
            self.state[function_index][len(self.function_list)] = module_index
            for i in range(len(self.function_list)):
                temp_edge = function + " -> " + self.function_list[i]
                if(temp_edge in self.edge_list):
                    self.state[function_index][i] = 1
        
        #print(self.state)
        # check the system complexity:
        self.old_complexity = 0

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
            if(complexity_nodes == 0):
                #print("module " + str(i) + " is skipped because of no node")
                continue
            
            #print("module " + str(i) + " has " + str(complexity_nodes) + \
            #      " nodes and " + str(complexity_edges) + " edges")
        
            #print(module_complexity)
            self.old_complexity += module_complexity  
            
        if(reset == False):
            print("Game info: ")
            print("The size of state matrix is [" + str(h) + "," + str(w) + "]")
            print("There are " + str(h) + " functions")
            print("The original complexity is " + str(self.old_complexity))
        
    # update the state matrix 
    def get_new_state(self, first_node_index, second_node_index):
          
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
                if(i == second_node_index and self.state[i][j] == 1):
                    self.state[i][j] = 0
                    self.state[first_node_index][j] = 1
            
                # then consider second node's calling
                if(j == second_node_index and self.state[i][j] == 1):
                    self.state[i][j] = 0
                    self.state[i][first_node_index] = 1
                   
        done = True
        node_count = 0
        
        for i in range(len(self.function_list)):
            if(self.state[i][len(self.function_list)] != -1):
                node_count += 1
                
        if(node_count >= 1):
            done = False
                    
        return done
                    
        print(self.state)
        
    # update the state matrix last element in each row
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
    
    


