#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:36:22 2019

agent_helper.py: This is the agent helper function, 
                 used to help agent act during training

@author: weijianzheng
"""

import random
import numpy as np

# the function to return the two nodes needs to be merged randomly
def find_two_nodes_random(state):
        
    num_functions = len(state)
    
    merge_node_index = random.sample(range(0, num_functions-1), 2)
    first_node_index = merge_node_index[0]
    second_node_index = merge_node_index[1]
    #print("first select two nodes " + str(first_node_index) + " and " \
    # + str(second_node_index))

    while(state[first_node_index][num_functions] == -1 or \
          state[second_node_index][num_functions] == -1):
        merge_node_index = random.sample(range(0, num_functions), 2)
        first_node_index = merge_node_index[0]
        second_node_index = merge_node_index[1]
            
#         print("then select two nodes " + str(first_node_index) + " and " \
#           + str(second_node_index))

    #print("Two selected nodes are " + str(first_node_index) + \
    #        " and " + str(second_node_index))
    
    return (first_node_index, second_node_index) 
    

# the function to return the two nodes needs to be merged
def find_two_nodes(state, A2):
    
    num_functions = len(state)
        
    temp_action_array = A2.copy()
    temp_action_array = np.asarray(temp_action_array)
    #print(temp_action_array.shape)
        
    first_node = np.amax(temp_action_array)
    first_node_index_array = np.argwhere(temp_action_array == first_node)
    first_node_index = first_node_index_array[0][2]
    #print(first_node_index)    
        
    while(state[first_node_index][num_functions] == -1):
            
        if(temp_action_array.size > 1):
            index_array = np.argwhere(temp_action_array == first_node)
            temp_action_array = np.delete(temp_action_array, index_array)
            first_node = np.amax(temp_action_array)
            first_node_index_array = np.argwhere(np.asarray(A2) == first_node)
            first_node_index = first_node_index_array[0][2]
            
    #print(temp_action_array)
    
    # find the content for the first node
    first_node = np.amax(temp_action_array)
    
    # this is used to find the index of the largest array
    first_node_index_array = np.argwhere(np.asarray(A2) == first_node)
    first_node_index = first_node_index_array[0][2]
    
    #print(first_node_index)
            
    #print(temp_action_array)
    index_array = np.argwhere(temp_action_array == first_node)
    index = index_array#[0][2]
    temp_action_array = np.delete(temp_action_array, index)
        
    second_node = np.amax(temp_action_array)
    second_node_index_array = np.argwhere(np.asarray(A2) == second_node)
    second_node_index = second_node_index_array[0][2]
    
    #print(second_node_index)
    #print(state)

    while(state[second_node_index][num_functions] == -1):
        
        # now need to remove one with -1
        if(temp_action_array.size > 1):
            index_array = np.argwhere(temp_action_array == second_node)
            #print(temp_action_array)
            temp_action_array = np.delete(temp_action_array, index_array)
            #print("delete " + str(index_array))
            second_node = np.amax(temp_action_array)
            second_node_index_array = np.argwhere(np.asarray(A2)==second_node)
            second_node_index = second_node_index_array[0][2]
        
        #if(temp_action_array.size == 1):
        #    break
        
        #print(temp_action_array)

    second_node = np.amax(temp_action_array)
    second_node_index_array = np.argwhere(np.asarray(A2) == second_node)
    second_node_index = second_node_index_array[0][2]
    #print(second_node_index)
        
    #print("Two selected nodes are " + str(first_node_index) + \
    #        " and " + str(second_node_index))
    
    return (first_node_index, second_node_index)    


# the function to return the two nodes needs to be merged
def find_one_node(state, A2):
        
    temp_action_array = A2.copy()
    temp_action_array = np.asarray(temp_action_array)
    #print(temp_action_array.shape)
        
    first_node = np.amax(temp_action_array)
    first_node_index_array = np.argwhere(temp_action_array == first_node)
    first_node_index = first_node_index_array[0][2]
    #print(first_node_index)    
    
    return (first_node_index)    