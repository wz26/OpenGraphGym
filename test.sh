#!/bin/sh

#sequential part of test cases:

# accuracy test
# test case for 20-20 sequential inference 
horovodrun -np 1 python sequential_inference.py 128 20 20
# test case for 20-50 batched inference 
horovodrun -np 1 python batched_inference.py 128 20 50 
# test case for 50-50 
horovodrun -np 1 python batched_inference.py 128 50 50

# parameter test
# test case for 20-20 batch size 64
horovodrun -np 1 python batched_inference.py 64 20 20 
# test case for 20-20 batch size 128
horovodrun -np 1 python batched_inference.py 128 20 20
# test case for 20-20 batch size 256
horovodrun -np 1 python batched_inference.py 256 20 20
# test case for 20-20 batch size 512
horovodrun -np 1 python batched_inference.py 512 20 20
# test case for 20-20 batch size 1024
horovodrun -np 1 python batched_inference.py 1024 20 20

# generalization test
# test case for 20-50 
horovodrun -np 1 python batched_inference.py 128 20 50 


#multiple GPUs test:
# 2 GPUs:
horovodrun -np 2 python batched_inference_2p.py 128 20 20
# 4 GPUs
horovodrun -np 4 python batched_inference_4p.py 128 20 20