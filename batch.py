import os
import time
import random
from subprocess import Popen, check_call
from GPUInfo import GPUInfo

gpu_usable = [0,1,2]
dataset = "/mv_users/peiguo/dataset/cub-fewshot/full/"
nclasses = 200
poolings = ["GAP", "GMP", "KMP", "LPP", "SMP", "MXP", "GTP", "STP", "LAEP"]

tasks = []
task_template = "python main.py --pretrained {} --nclasses {} --pool_name {} --lr 0.001 30 0.0001 20"
for pool in poolings:
   tasks.append(task_template.format(dataset, nclasses, pool)) 

while len(tasks) > 0:
    gpu_info = GPUInfo.info()
    for gpu_id in gpu_usable:
        util = gpu_info[gpu_id]["gpu_util"]
        mem_used = gpu_info[gpu_id]["mem_used"]
        print("*"*30)
        print("gpu_id: {}, util: {}, mem used: {}".format(gpu_id, util, mem_used))
        print("*"*30)
        if len(tasks) == 0:
            break
        if mem_used > 1000:
            continue
        cmdstr = "CUDA_VISIBLE_DEVICES={} ".format(gpu_id) + tasks[0]
        Popen(cmdstr, shell=True)
        del tasks[0]
        time.sleep(10)

    print("All GPUs are utilized, check back in 10 mins")
    time.sleep(600)

print("*"*30)
print('All tasks are finished')
print("*"*30)
