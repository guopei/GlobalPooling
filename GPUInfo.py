"""
Parse nvidia-smi xml output to obtain gpu usage.
Work with batch.py as task scheduler.
Pei Guo
BYU
Aug, 2019
"""

import subprocess as sp
import xml.etree.ElementTree as et

class GPUInfo():
    def __init__(self):
        pass

    @staticmethod
    def info():
        gpu_info = []
        nvidia_smi = sp.check_output('nvidia-smi -x -q', shell=True)
        nvidia_smi = et.fromstring(nvidia_smi)
        
        for gpu_tag in nvidia_smi.iter("gpu"):
            gpu_util = gpu_tag.find('utilization').find('gpu_util').text.split()[0]
            mem_used = gpu_tag.find('fb_memory_usage').find('used').text.split()[0]
            gpu_info.append({
                'gpu_util': float(gpu_util), 
                'mem_used': float(mem_used)
                })

        return gpu_info

if __name__ == "__main__":
    info = GPUInfo()
    print(info.info())


