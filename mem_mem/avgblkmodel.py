from math import *
import pandas as pd
import numpy as np

class DeviceInfo():
    def __init__(self, sm_num=0, sharedmem_per_sm=0, reg_per_sm=0, maxthreads_per_sm=0):
        self.sm_num = sm_num
        self.sharedmem_per_sm = sharedmem_per_sm # bytes
        self.reg_per_sm = reg_per_sm
        self.maxthreads_per_sm = maxthreads_per_sm


class KernelInfo():
    def __init__(self, blockDim=0, gridDim=0, reg_per_thread=0, sharedmem_per_blk=0, runtime_ms = 0):
        self.blockDim = blockDim
        self.gridDim = gridDim
        self.reg_per_thread = reg_per_thread
        self.sharedmem_per_blk =  sharedmem_per_blk
        self.runtime_ms = runtime_ms


def MaxBLK_Per_SM(Gpu, Kern):
    """
    Compute the max blocks on one SM
    """
    warp_size = 32
    DeviceLimit = Gpu.maxthreads_per_sm / 32

    blocks_by_sm = DeviceLimit

    if Kern.sharedmem_per_blk > 0:
        blocks_by_sm = floor(Gpu.sharedmem_per_sm / float(Kern.sharedmem_per_blk)) # int operation

    blocks_by_reg = floor(Gpu.reg_per_sm / float(Kern.reg_per_thread * Kern.blockDim))

    blocks_by_threads = floor(Gpu.maxthreads_per_sm / float(Kern.blockDim))

    # maxblks_per_sm
    return min([blocks_by_sm, blocks_by_reg, blocks_by_threads])


def compute_avgblktime(Gpu, kernel):
    max_blk_per_sm = MaxBLK_Per_SM(Gpu, kernel)
    #print('max blk per sm = {}'.format(max_blk_per_sm))

    # max blocks that can be launhed on gpu at once time
    # if there are more blocks, they will wait for the next iteration
    # each SM starts and finishes at the same time
    # all the blocks on that SM starts and ends at the same time
    block_per_iteration = Gpu.sm_num * max_blk_per_sm

    iterations = ceil(kernel.gridDim / block_per_iteration) # total iterations
    #print 'iterations ' + str(iterations)

    # divide the kernel runtime by the number of iterations will be the avg block exeuction time for our model
    avg_blk_time = kernel.runtime_ms / float(iterations)
    #print('avg block execution time (ms) : {}'.format(avg_blk_time))

    return avg_blk_time

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class sm_stat:
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


def check_sm_resource(current_sm, block_info):
    enough_thread = current_sm.thread >= block_info.blockDim
    enough_reg = current_sm.reg >= (block_info.reg_per_thread * block_info.blockDim)
    enough_sm = current_sm.sharedmem >= block_info.sharedmem_per_blk

    allocate = 0
    if enough_thread and enough_reg and enough_sm:
        allocate = 1

    return allocate


def Search_block_start(df_sm_trace, current_kernel_id):
    """
    Read the sm_trace table, find out all the active blocks on current sm, look for the earliest start
    """
    df_active = df_sm_trace.loc[df_sm_trace['active'] == 1]
    blk2start = df_active['block_start'].max() # find the closest block

    df_active_current_kernel = df_active.loc[df_active['kernel_id'] == current_kernel_id]
    if not df_active_current_kernel.empty:
        blk2start = df_active_current_kernel['block_start'].max()  # find the closest blk for current kernel

    return blk2start
