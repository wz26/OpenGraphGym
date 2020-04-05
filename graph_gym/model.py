#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:49:09 2019

@author: weijianzheng
"""

import tensorflow as tf
import numpy as np
from collections import deque
import copy
import horovod.tensorflow as hvd

class Model:
    
    def __init__(self, model_info):
        
        # Horovod: initialize Horovod.
        print(hvd.size())
        
        # first need to parse the application info and define nn
        self.game_name = model_info[0]

        # set remaining variables
        self.batch_size = 128
        self.learning_rate = model_info[5]

        if(self.game_name == 'min_cover_s2v'):            
            self.num_nodes = model_info[1]
            self.embed_dim = model_info[2]
            
            self.epochs = 50
            ### define placeholder as input
            # input of the model (X decided by selected or not)
            self.x  = tf.placeholder(shape=[None,None], dtype=tf.float32)
            self.ad_matrix = tf.placeholder(shape=[None,None,None], \
                    dtype=tf.float32)

            # label value
            self.y  = tf.placeholder(shape=[1, None],dtype=tf.float32)
           
            # used to obtain mu_v
            self.select_vec = tf.placeholder(shape=[None, None, 1], \
                    dtype=tf.float32)
            # used to obtain sigma mu_u
            self.all_vec    = tf.placeholder(shape=[None, None, 1], \
                    dtype=tf.float32)

            self.init_limit = 0.5
            self.embed_layer = 5
            self.hid_num_units = 16

            ### define parameter for model of embedding 
            self.embed_param = {
                'theta1':tf.Variable(tf.random_uniform([self.embed_dim, 1], \
                                                       minval=-self.init_limit, \
                                                       maxval=self.init_limit, \
                                                       dtype=tf.float32, seed=1)),
                'theta2':tf.Variable(tf.random_uniform([self.embed_dim, self.embed_dim], \
                                                       minval=-self.init_limit, \
                                                       maxval=self.init_limit, \
                                                       dtype=tf.float32, seed=1)),
                'theta3':tf.Variable(tf.random_uniform([self.embed_dim, self.embed_dim], \
                                                       minval=-self.init_limit, \
                                                       maxval=self.init_limit, \
                                                       dtype=tf.float32, seed=1)),
                'theta4':tf.Variable(tf.random_uniform([self.embed_dim, 1], \
                                                       minval=-self.init_limit, \
                                                       maxval=self.init_limit, \
                                                       dtype=tf.float32, seed=1))
            }

            ### define parameter for model of Q
            self.value_param = {
                'hid':tf.Variable(tf.random_uniform([2*self.embed_dim, self.hid_num_units], \
                                                    minval=-self.init_limit, \
                                                    maxval=self.init_limit, \
                                                    dtype=tf.float32, seed=1)),
                'theta5':tf.Variable(tf.random_uniform([self.hid_num_units, 1], \
                                                       minval=-self.init_limit, \
                                                       maxval=self.init_limit, \
                                                       dtype=tf.float32, seed=1)),
                'theta6':tf.Variable(tf.random_uniform([self.embed_dim, self.embed_dim], \
                                                       minval=-self.init_limit, \
                                                       maxval=self.init_limit, \
                                                       dtype=tf.float32, seed=1)),
                'theta7':tf.Variable(tf.random_uniform([self.embed_dim, self.embed_dim], \
                                                       minval=-self.init_limit, \
                                                       maxval=self.init_limit, \
                                                       dtype=tf.float32, seed=1))
            }

            self.size =  tf.shape(self.ad_matrix)[0]
            self.size_nodes =  tf.shape(self.ad_matrix)[1]            
            ### first level of embedding:
            self.theta4_relu = tf.nn.relu(tf.tile(self.embed_param['theta4'], \
                    tf.stack([1, self.size_nodes])))
            ## tf.broadcast_to only supported after 1.13.0
            #self.mu_0_sigma  = tf.matmul(self.embed_param['theta3'], \
            #        tf.matmul(self.theta4_relu, self.ad_matrix))
            #self.theta4_relu = tf.reshape(self.theta4_relu, [-1, self.embed_dim, self.num_nodes])
            self.theta4_relu = tf.expand_dims(self.theta4_relu, 0)
            self.theta4_relu = tf.tile(self.theta4_relu, [self.size, 1, 1], name=None)
            #self.theta4_relu = tf.broadcast_to(self.theta4_relu, [self.size, self.embed_dim, self.num_nodes])
            #self.theta4_relu = tf.expand_dims(self.theta4_relu, 0)
            #self.theta4_relu = tf.stack([x, y, z], axis=0)   self.size
            self.theta3 = tf.expand_dims(self.embed_param['theta3'], 0)
            self.theta3 = tf.tile(self.theta3, [self.size, 1, 1], name=None)
            #self.theta3 = tf.broadcast_to(self.theta3, [self.size, self.embed_dim, self.embed_dim])
           
            self.mu_0_sigma  = tf.matmul(self.theta3, tf.matmul(self.theta4_relu, self.ad_matrix))
            
#             # now our output is Bn-by-p
            self.theta1 = tf.expand_dims(self.embed_param['theta1'], 0)
            self.theta1 = tf.tile(self.theta1, [self.size, 1, 1], name=None)
            self.x_reshape = tf.expand_dims(self.x, 0)
            self.x_reshape = tf.transpose(self.x_reshape, perm=[1, 0, 2])
            self.mu_0 = tf.nn.relu(tf.add(tf.matmul(self.theta1, self.x_reshape), self.mu_0_sigma))
           
            ### second level of embedding:
            self.theta2 = tf.expand_dims(self.embed_param['theta2'], 0)
            self.theta2 = tf.tile(self.theta2, [self.size, 1, 1], name=None)
            self.mu_1_sigma  = tf.matmul(self.theta2, tf.matmul(self.mu_0, self.ad_matrix))
            self.mu_1        = tf.add(tf.matmul(self.theta1, self.x_reshape), self.mu_1_sigma)
            self.mu_1        = tf.nn.relu(tf.add(self.mu_1, self.mu_0_sigma))
            # now bn-by-p
            for i in range(2, self.embed_layer):
                ### second level of embedding:
                self.mu_1_sigma  = tf.matmul(self.theta2, tf.matmul(self.mu_1, self.ad_matrix))
                self.mu_1        = tf.add(tf.matmul(self.theta1, self.x_reshape), self.mu_1_sigma)
                self.mu_1        = tf.nn.relu(tf.add(self.mu_1, self.mu_0_sigma))

            ### Q value function
            self.Q_sigma = tf.matmul(self.mu_1, self.all_vec)

            self.theta6 = tf.expand_dims(self.value_param['theta6'], 0)
            self.theta6 = tf.tile(self.theta6, [self.size, 1, 1], name=None)
            self.Q_sigma = tf.matmul(self.theta6, self.Q_sigma)
        
            self.mu_u = tf.matmul(self.mu_1, self.select_vec)
            
            self.theta7 = tf.expand_dims(self.value_param['theta7'], 0)
            self.theta7 = tf.tile(self.theta7, [self.size, 1, 1], name=None)
            self.mu_u = tf.matmul(self.theta7, self.mu_u)  

            self.Q_vec        = tf.nn.relu(tf.concat([self.Q_sigma, self.mu_u], 1))
            
            self.hid = tf.expand_dims(tf.transpose(self.value_param['hid']), 0)
            self.hid = tf.tile(self.hid, [self.size, 1, 1], name=None)
            self.output_hidden = tf.matmul(self.hid, self.Q_vec)            
            
            self.theta5 = tf.expand_dims(tf.transpose(self.value_param['theta5']), 0)
            self.theta5 = tf.tile(self.theta5, [self.size, 1, 1], name=None)
            self.output_layer = tf.matmul(self.theta5, self.output_hidden)
        else:
            self.epochs = 3
            self.in_num_units = model_info[1]
            self.hid_num_units = model_info[2]
            self.out_num_units = model_info[3]   

            # define placeholders
            self.x = tf.placeholder(shape=[None,self.in_num_units], dtype=tf.float32)
            self.y = tf.placeholder(shape=[None,self.out_num_units],dtype=tf.float32)

            ### define weights and biases of the neural network 
            self.weights = {
                'hidden1':tf.Variable(tf.random_normal([self.in_num_units, \
                                                        self.hid_num_units \
                                                        ], seed=1)),
                'hidden2':tf.Variable(tf.random_normal([self.hid_num_units, \
                                                        int(self.hid_num_units/2) \
                                                        ], seed=1)),
                'output':tf.Variable(tf.random_normal([int(self.hid_num_units/2), \
                                                        self.out_num_units \
                                                        ], seed=1))
            }

            self.biases = {
                'hidden1':tf.Variable(tf.random_normal([self.hid_num_units], seed=1)),
                'hidden2':tf.Variable(tf.random_normal([int(self.hid_num_units/2)],\
                        seed=1)),
                'output':tf.Variable(tf.random_normal([self.out_num_units], seed=1))
            }

            self.hidden_layer1 = tf.add(tf.matmul(self.x, self.weights['hidden1']),\
                                       self.biases['hidden1'])
            self.hidden_layer1 = tf.nn.relu(self.hidden_layer1)

            self.hidden_layer2 = tf.add(tf.matmul(self.hidden_layer1, \
                                                  self.weights['hidden2']), \
                                                  self.biases['hidden2'])
            self.hidden_layer2 = tf.nn.relu(self.hidden_layer2)

            self.output_layer = tf.matmul(self.hidden_layer2, \
                                          self.weights['output']) \
                                          + self.biases['output']
            # self.output_layer = tf.nn.relu(self.output_layer)

        self.cost = tf.reduce_sum(tf.square(self.output_layer - self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        global_step = tf.train.get_or_create_global_step()
    
        self.hvd_opt = hvd.DistributedOptimizer(self.optimizer)
        self.train_op = self.hvd_opt.minimize(self.cost, global_step=global_step)
        
        #global_step = tf.Variable(0, trainable=False)
        
        #self.train_op = self.hvd_opt.minimize(self.cost, global_step=global_step)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 1
        gpus = config.gpu_options.visible_device_list= str(hvd.local_rank())
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        #gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        
        #config.gpu_options.visible_device_list = str(hvd.local_rank())
        print(config.gpu_options.visible_device_list)
        
        self.sess = tf.Session(config=config)
        
        self.sess.run(tf.global_variables_initializer())
        
        bcast = hvd.broadcast_global_variables(0)
        
        print("it has been initialized")

        self.hooks = [
            # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
            # from rank 0 to all other processes. This is necessary to ensure consistent
            # initialization of all workers when training is started with random weights
            # or restored from a checkpoint.
            hvd.BroadcastGlobalVariablesHook(0)

            # Horovod: adjust number of steps based on number of GPUs.
            #tf.train.StopAtStepHook(last_step=self.epochs // hvd.size()),

#             tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': self.cost},
#                                        every_n_iter=10),
        ]

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
       
        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting them.
        self.checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
#         training_batch_generator = train_input_generator(x_train,
#                                                          y_train, batch_size=100)
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
#         with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
#                                                hooks=self.hooks,
#                                                config=config) as mon_sess:
#             while not mon_sess.should_stop():
#                 # Run a training step synchronously.
#                 image_, label_ = next(training_batch_generator)
#                 mon_sess.run(train_op, feed_dict={image: image_, label: label_})
        self.mon_sess = tf.train.MonitoredTrainingSession(hooks=self.hooks, config=config)

#         self.mon_sess = tf.train.MonitoredTrainingSession(config=config)
        #self.init = tf.initialize_all_variables()    
        #self.sess = tf.Session(target=model_info[4], config=config)
        
        #self.sess.run(tf.global_variables_initializer())

        #self.saver = tf.train.Saver()
        
        #self.sess.graph.finalize()

    def forward(self, state):
        if(self.game_name == 'min_cover_s2v'):
            #with tf.device("/gpu:{}".format(0)):
            self.A2 = self.mon_sess.run([self.output_layer],feed_dict={self.x:state[0], \
                               self.ad_matrix:state[1], self.select_vec:state[2], \
                               self.all_vec:state[3]})
        else:
            self.A2 = self.sess.run([self.output_layer],feed_dict={self.x:state.\
                                    reshape((1, self.in_num_units))})
            
    def forward_batch(self, state):
        if(self.game_name == 'min_cover_s2v'):
            #with tf.device("/gpu:{}".format(0)):
            self.A2 = self.mon_sess.run([self.output_layer],feed_dict={self.x:state[0], \
                               self.ad_matrix:state[1], self.select_vec:state[2], \
                               self.all_vec:state[3]})
            #print(np.squeeze(self.size))
        else:
            self.A2 = self.sess.run([self.output_layer],feed_dict={self.x:state.\
                                    reshape((1, self.in_num_units))})

    def backward(self, state, label):
        #with tf.device("/gpu:{}".format(0)):
        if(self.game_name == 'min_cover_s2v'):
            #print(state)
            #label_test = 1
            #np.zeros(shape = (1, 1))
            new_cost = 0
            old_cost = 0
            i = 0
            while(new_cost <= old_cost):
            #for i in range(0, self.epochs):
#                 _,c = self.sess.run([self.optimizer, self.cost], feed_dict={\
#                         self.x:state[0], self.ad_matrix:state[1], \
#                         self.select_vec:state[2], \
#                         self.all_vec:state[3], \
#                         self.y:label})
                _,c = self.mon_sess.run([self.optimizer, self.cost], feed_dict={\
                        self.x:state[0], self.ad_matrix:state[1], \
                        self.select_vec:state[2], \
                        self.all_vec:state[3], \
                        self.y:label})
                old_cost = new_cost
                new_cost = c
                if(i == 0): 
                    old_cost = new_cost
                i = i+1
        else:
            for i in range(0, self.epochs):
                _,c = self.sess.run([self.optimizer, self.cost], feed_dict = {\
                                    self.x:state.reshape((1, self.in_num_units)),\
                                    self.y:label.reshape((1, self.out_num_units))})
                
    def backward_batch(self, exps, batch_size):
        #with tf.device("/gpu:{}".format(0)):
        #print(exps)
        states_x = np.zeros(shape = (batch_size, self.num_nodes))
        states_ad_matrix = np.zeros(shape = (batch_size, self.num_nodes, self.num_nodes))
        states_select_vec = np.zeros(shape = (batch_size, self.num_nodes, 1))
        states_all_vec = np.ones(shape = (batch_size, self.num_nodes, 1))
        labels = np.zeros(shape = (1, batch_size))
        
        if(batch_size == 1):
            states_x[0] = copy.deepcopy(exps[0][0])
            states_ad_matrix[0] = copy.deepcopy(exps[0][1])
            states_select_vec[0] = copy.deepcopy(exps[0][2])
            states_all_vec[0] = copy.deepcopy(exps[0][3])
            labels[0][0] = copy.deepcopy(exps[1])
        else:
            for i in range(batch_size):
                states_x[i] = copy.deepcopy(exps[i][0][0])
                states_ad_matrix[i] = copy.deepcopy(exps[i][0][1])
                states_select_vec[i] = copy.deepcopy(exps[i][0][2])
                states_all_vec[i] = copy.deepcopy(exps[i][0][3])
                labels[0][i] = copy.deepcopy(exps[i][1])
            
        
        if(self.game_name == 'min_cover_s2v'):

            new_cost = 0
            old_cost = 0
            i = 0
            #while(new_cost <= old_cost):
            #bcast = hvd.broadcast_global_variables(0)
            
            for i in range(0, self.epochs):
                 #feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                 # We perform one update step by evaluating the optimizer op (including it
                    # in the list of returned values for session.run()
                 #_, loss_val = session.run([train_op, loss], feed_dict=feed_dict)
                
                 _,c = self.mon_sess.run([self.train_op, self.cost], feed_dict={\
                            self.x:states_x, self.ad_matrix:states_ad_matrix, \
                            self.select_vec:states_select_vec, \
                            self.all_vec:states_all_vec, \
                            self.y:labels})
#                  print(c)
                 old_cost = new_cost
                 new_cost = c
# #                  if(i == self.epochs-1): 
# #                     old_cost = new_cost
# #                     print(c)
# #                  i = i+1
                 if(i == 0): 
                    old_cost = new_cost
                 if(new_cost > old_cost):
# #                     print(i)
# #                     print(c)
                    break
        else:
            for i in range(0, self.epochs):
                _,c = self.sess.run([self.optimizer, self.cost], feed_dict = {\
                                    self.x:state.reshape((batch_size, self.in_num_units)),\
                                    self.y:label.reshape((batch_size, self.out_num_units))})

    def save(self):
        
#         self.save_path = self.saver.save(self.sess, \
#                                          "../model_saved/model.ckpt")
        print("Model saved in path: %s" % self.save_path)
        
        return True
        
    def load(self):
        
        self.sess = tf.Session()

        self.saver.restore(self.sess, "../model_saved/model.ckpt")
        print("Model restored.")
        
        return True  
