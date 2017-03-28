import pandas as pd
import numpy as np
from math import *


class DeviceInfo():
    def __init__(self, sm_num=0, sharedmem_per_sm=0, reg_per_sm=0, maxthreads_per_sm=0):
        self.sm_num = sm_num
        self.sharedmem_per_sm = sharedmem_per_sm # bytes
        self.reg_per_sm = reg_per_sm
        self.maxthreads_per_sm = maxthreads_per_sm


class KernelInfo():
    def __init__(self, blockDim=0, gridDim=0, reg_per_thread=0, sharedmem_per_blk=0, 
                 runtime_ms = 0, avg_blk_time = 0, start = 0):
        self.blockDim = blockDim
        self.gridDim = gridDim
        self.reg_per_thread = reg_per_thread
        self.sharedmem_per_blk =  sharedmem_per_blk
        self.runtime_ms = runtime_ms
        self.avg_blk_time = avg_blk_time
        self.start_ms = start


class KernConfig():
    def __init__(self,
                 grid_x = 0, grid_y = 0, grid_z = 0,
                 blk_x = 0, blk_y = 0, blk_z = 0,
                 regs_per_thread = 0, sm_per_block = 0):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.blk_x = blk_x
        self.blk_y = blk_y
        self.blk_z = blk_z
        self.regs_per_thread = regs_per_thread
        self.sm_per_block = sm_per_block

#-------------------------------------------------------------------------------
# SM status
#-------------------------------------------------------------------------------
class SM_Stat:
    def __init__(self, thread=0, reg=0, sharedmem = 0, full=0, batch = 1):
        self.thread = thread
        self.reg= reg
        self.sharedmem = sharedmem
        self.full = full
        self.batch = batch

    def init(self, Gpu):
        self.thread = Gpu.maxthreads_per_sm
        self.reg = Gpu.reg_per_sm
        self.sharedmem = Gpu.sharedmem_per_sm
        self.full = 0
        self.batch = 1

    def replenish(self, Gpu):
        self.thread = Gpu.maxthreads_per_sm
        self.reg = Gpu.reg_per_sm
        self.sharedmem = Gpu.sharedmem_per_sm
        self.full = 0
        self.batch += 1 # add

    def Rm(self, Kern):
        """
        Remove the kernel block occupied resource by adding back them.
        """
        self.thread += Kern.blockDim
        self.reg += Kern.reg_per_thread * Kern.blockDim
        self.sharedmem += Kern.sharedmem_per_blk

    def Allocate_block(self, Kern):
        self.thread -= Kern.blockDim
        self.reg -= Kern.reg_per_thread * Kern.blockDim
        self.sharedmem -= Kern.sharedmem_per_blk


#------------------------------------------------------------------------------
# Compute max blocks per sm
#------------------------------------------------------------------------------
def MaxBLK_Per_SM(Gpu, Kern):
    """
    Compute the max blocks on one SM
    """
    warp_size = 32
    DeviceLimit = Gpu.maxthreads_per_sm / warp_size

    blocks_by_sm = DeviceLimit

    if Kern.sharedmem_per_blk > 0:
        blocks_by_sm = floor(Gpu.sharedmem_per_sm / float(Kern.sharedmem_per_blk)) # int operation

    blocks_by_reg = floor(Gpu.reg_per_sm / float(Kern.reg_per_thread * Kern.blockDim))

    blocks_by_threads = floor(Gpu.maxthreads_per_sm / float(Kern.blockDim))

    # maxblks_per_sm
    return min([blocks_by_sm, blocks_by_reg, blocks_by_threads])


#------------------------------------------------------------------------------
# Compute Average block execution time. 
#------------------------------------------------------------------------------
def AvgBlkTime(Gpu, kernel):
    max_blk_per_sm = MaxBLK_Per_SM(Gpu, kernel)

    # max blocks that can be launhed on gpu at once time
    # if there are more blocks, they will wait for the next iteration
    # each SM starts and finishes at the same time
    # all the blocks on that SM starts and ends at the same time
    block_per_iteration = Gpu.sm_num * max_blk_per_sm

    iterations = ceil(kernel.gridDim / block_per_iteration) # total iterations
    #print 'iterations ' + str(iterations)

    # divide the kernel runtime by the number of iterations will be the avg 
    # block exeuction time for our model
    avg_blk_time = kernel.runtime_ms / float(iterations)
    #print('avg block execution time (ms) : {}'.format(avg_blk_time))

    return avg_blk_time


#------------------------------------------------------------------------------
# Check whether there is enough resource to allocate a block
#------------------------------------------------------------------------------
def check_sm_resource(current_sm, block_info):
    enough_thread = current_sm.thread >= block_info.blockDim
    enough_reg = current_sm.reg >= (block_info.reg_per_thread * block_info.blockDim)
    enough_sm = current_sm.sharedmem >= block_info.sharedmem_per_blk
    
    allocate = False
    if enough_thread and enough_reg and enough_sm:
        allocate = True
    
    return allocate


#------------------------------------------------------------------------------
# Init the gpu resouce and trace table for each SM 
#------------------------------------------------------------------------------
def init_gpu(Gpu):
    #------------------
    # init SM resources
    #------------------
    sm_num = Gpu.sm_num
    sm_res_list = [SM_Stat() for i in range(sm_num)] # a list of sm resources
    for i in range(sm_num):
        sm_res_list[i].init(Gpu)

    #----------------------------------------------------------------
    # a trace table to record all the block trace: using pd dataframe
    #----------------------------------------------------------------
    trace_table = pd.DataFrame(columns=['sm_id', 'block_id', 'block_start', 
        'block_end', 'batch_id', 'kernel_id', 'active'])
    # have a trace table for each sm
    sm_trace = [trace_table for x in range(sm_num)]

    return sm_res_list, sm_trace


#------------------------------------------------------------------------------
# Copy current kernel info to another data structure 
# like deep copy
#------------------------------------------------------------------------------
def Copy_kern_info(kernel):
    kern = KernelInfo(kernel.blockDim,
            kernel.gridDim,
            kernel.reg_per_thread,
            kernel.sharedmem_per_blk,
            kernel.runtime_ms,
            kernel.avg_blk_time,
            kernel.start_ms)
    return kern
