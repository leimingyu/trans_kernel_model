import pandas as pd
import numpy as np
from math import *
import operator


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
# Update Cell Function.
#------------------------------------------------------------------------------
def UpdateCell(df_all_api, row_id, col_name, val):
    df = df_all_api.copy(deep=True)
    df.set_value(row_id, col_name, val)
    return df


#------------------------------------------------------------------------------
# Set the target row to be wake status, return stream id
#------------------------------------------------------------------------------
def SetWake(df_all, r1):
    df= df_all.copy(deep=True)
    df= UpdateCell(df, r1, 'status', 'wake')
    # also update the current and pred end
    df = UpdateCell(df, r1, 'current_pos', get_rowinfo(df, r1)['start'])
    df = UpdateCell(df, r1, 'pred_end', get_rowinfo(df, r1)['end'])
    return df


#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def SortKern(df_all, kernlist):
    ken_num = len(kernlist)
    if ken_num == 1:
        return kernlist

    sorted_kern = [] 

    # sort by the kernel starting time
    k_dd = {}
    for k_row in kernlist:
        k_dd[k_row] = GetInfo(df_all, k_row, 'start')

    sorted_kern = sorted(k_dd.items(), key=operator.itemgetter(1))

    kern_row_list_sort = []
    for item in sorted_kern:
        kern_row_list_sort.append(item[0])

    return kern_row_list_sort 


#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def GetKernID(df, mystream, myrow):
    df_mystream = df.loc[df.stream_id == mystream]

    count = 0
    kern_id = None
    for index, row in df_mystream.iterrows():
        if row.api_type == 'kern': 
            if index == myrow:
                kern_id = count 
            else:
                count += 1

    return kern_id

#------------------------------------------------------------------------------
# GetKernelInfoandTag 
#------------------------------------------------------------------------------
def GetKernelInfoAndTag(df, row, stream_kernel_list):
    mystream = GetInfo(df, row, 'stream_id')
    mystream = int(mystream)

    my_kernid = GetKernID(df, mystream, row)
    #print('kern row {}, stream {}, kern_id_in_stream {}'.format(row, mystream, my_kernid))

    my_kernel_info = stream_kernel_list[mystream][my_kernid]
    my_kernel_info.start_ms = GetInfo(df, row, 'start')
    #
    # kernel id label
    kid = GetInfo(df, row, 'kern_id')

    return my_kernel_info, kid


#------------------------------------------------------------------------------
# Find Kernel Record for sm trace 
#------------------------------------------------------------------------------
def FindKernelRecord(SMtracelist, kid):
    Found = False
    for df_sm in SMtracelist:
        kernid_list = df_sm['kernel_id'].unique()
        if kid in kernid_list:
            Found = True
            break
    return Found

#------------------------------------------------------------------------------
# Set the target row to be wake status, return stream id
#------------------------------------------------------------------------------
def GetWakeListByTime(df_all, begT, endT):
    df= df_all.copy(deep=True)
    df_wake = df.loc[df.status == 'wake']
    wake_list = []
    for index, row in df_wake.iterrows():
        if row.pred_end >= endT and row.start <= begT:
            wake_list.append(index)
    return wake_list


def GetWakeListBefore(df_all, endT):
    df= df_all.copy(deep=True)
    df_wake = df.loc[df.status == 'wake']

    wake_list = []
    for index, row in df_wake.iterrows():
        if row.start < endT:
            wake_list.append(index)
    return wake_list

#------------------------------------------------------------------------------
# Set the target row to be wake status, return stream id
#------------------------------------------------------------------------------
def FindOvlp(df_all, wakelist):
    df= df_all.copy(deep=True)
    h2d_list, d2h_list, kern_list = [],[],[]
    for row in wakelist:
        api_type = GetInfo(df, row, 'api_type')
        if api_type == 'h2d':  h2d_list.append(row)
        if api_type == 'd2h':  d2h_list.append(row)
        if api_type == 'kern': kern_list.append(row)
    return h2d_list, d2h_list, kern_list


#------------------------------------------------------------------------------
# Find wake list in the dataframe 
#------------------------------------------------------------------------------
def FindWakeList(df_all):
    df= df_all.copy(deep=True)
    df_wake = df.loc[df.status == 'wake']
    wake_list = [] 
    for index, row in df_wake.iterrows():
        wake_list.append(index)
    return wake_list 

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def Find_nextcall_samestream(df_all, row2nd, row2nd_stream): 
    df= df_all.copy(deep=True)

    df_stream = df.loc[df.stream_id == row2nd_stream]

    Next = False
    row = None
    for index, row in df_stream.iterrows():
        if Next:
            row = index
            break
        if index == row2nd:
            Next = True
    return row 


#------------------------------------------------------------------------------
# Compute_bytesleft_time
#------------------------------------------------------------------------------
def Compute_left_time(df, row, ways = 1.0):
    cc = float(ways)
    bytes_left = GetInfo(df, row, 'bytes_left')
    bw = GetInfo(df, row, 'bw')
    bw = bw / cc 
    trans_time =  bytes_left / bw
    return trans_time

#------------------------------------------------------------------------------
# Finish the target row 
#------------------------------------------------------------------------------
def Finish_row_h2d(df_all, row, simT, ways = 1.0):
    df = df_all.copy(deep=True)
    cc = float(ways)
    trans_time = Compute_left_time(df, row, ways = cc)
    new_end = simT + trans_time # new pred end
    tot_size = GetInfo(df, row, 'size_kb')
    # update info
    df = UpdateCell(df, row, 'bytes_done', tot_size) 
    df = UpdateCell(df, row, 'bytes_left', 0) 
    df = UpdateCell(df, row, 'current_pos', new_end) 
    df = UpdateCell(df, row, 'pred_end', new_end) 
    df = UpdateCell(df, row, 'status', 'done') 
    return df


#------------------------------------------------------------------------------
# simT to predendT, there is cc, after that cc = 1.0 
#------------------------------------------------------------------------------
def Update_row_h2d(df_all, row, simT, predendT, ways = 1.0):
    df = df_all.copy(deep=True)
    cc = float(ways)

    bw_org = GetInfo(df, row, 'bw')
    bw = bw_org / cc 

    bytes_left = GetInfo(df, row, 'bytes_left')
    bytes_done = GetInfo(df, row, 'bytes_done')

    bytes_transfer =  (predendT - simT) * bw

    bytes_left_new =  bytes_left  - bytes_transfer
    bytes_done_new =  bytes_left  + bytes_transfer


    time_left_no_ovlp = bytes_left_new / bw_org
    # predendT with cc + time left (transfer with org bw)
    pred_end_new =  predendT + time_left_no_ovlp

    # update info
    df = UpdateCell(df, row, 'bytes_done', bytes_done_new) 
    df = UpdateCell(df, row, 'bytes_left', bytes_left_new) 
    df = UpdateCell(df, row, 'current_pos', simT) 
    df = UpdateCell(df, row, 'pred_end', pred_end_new) 

    
    return df



#------------------------------------------------------------------------------
# update h2d api by the concurrency 
#------------------------------------------------------------------------------
def Update_h2d_bytes(df_all, row, startT, endT, ways = 1.0):
    df = df_all.copy(deep=True)
    cc = float(ways)

    bw = GetInfo(df, row, 'bw')
    bw = bw / cc 

    dur = endT - startT

    bytes_done = GetInfo(df, row, 'bytes_done') 
    bytes_left = GetInfo(df, row, 'bytes_left') 

    # compute bytes left
    bytes_tran = dur * bw 
    bytes_left_new = bytes_left - bytes_tran
    bytes_done_new = bytes_done + bytes_tran

    df = UpdateCell(df, row, 'bytes_done', bytes_done_new)
    df = UpdateCell(df, row, 'bytes_left', bytes_left_new)
    #df = UpdateCell(df, row, 'current_pos', endT)

    ## check whether ended
    #done = True if bytes_left_new <= 1e-3 else False 

    #if done:
    #    print('row to be done: row {}'.format(row))
    #else:
         

    return df


#------------------------------------------------------------------------------
# Find previous api call in the same stream 
#------------------------------------------------------------------------------
def Find_prevapi_samestream(df_all, r2, r2_stream):
    df = df_all.copy(deep=True)

    df_wake = df.loc[(df.status == 'wake') & (df.stream_id == r2_stream)]

    prev_call_list = []

    for index, _ in df_wake.iterrows():
        if index <> r2:
            prev_call_list.append(index) 

    if len(prev_call_list) <> 1:
        sys.stderr.write('something is wrong. check prev api call.') 

    return prev_call_list[0]


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
def Copy_kernel_info(kernel):
    kern = KernelInfo(kernel.blockDim,
            kernel.gridDim,
            kernel.reg_per_thread,
            kernel.sharedmem_per_blk,
            kernel.runtime_ms,
            kernel.avg_blk_time,
            kernel.start_ms)
    return kern


#------------------------------------------------------------------------------
# Print Kernel information 
#------------------------------------------------------------------------------
def Dump_kernel_info(kernel):
    print('Kernel Info'
        '\n\t\tblockDim {}'
        '\n\t\tgridkDim {}'
        '\n\t\tregs {}'
        '\n\t\tshared memory {}'
        '\n\t\truntime (ms) {}'
        '\n\t\taverage block execution time (ms) {}'
        '\n\t\tstart time (ms) {}'.format(kernel.blockDim,
        kernel.gridDim,
        kernel.reg_per_thread,
        kernel.sharedmem_per_blk,
        kernel.runtime_ms,
        kernel.avg_blk_time,
        kernel.start_ms))


def get_rowinfo(df_all, rowid):
    row_dd = {}
    row_dd['start'] = df_all.loc[rowid]['start']
    row_dd['end'] = df_all.loc[rowid]['end']
    row_dd['api_type'] = df_all.loc[rowid]['api_type']
    row_dd['size_kb'] = df_all.loc[rowid]['size_kb']
    row_dd['stream_id'] = df_all.loc[rowid]['stream_id']
    row_dd['status'] = df_all.loc[rowid]['status']
    row_dd['bw'] = df_all.loc[rowid]['bw']
    row_dd['bytes_done'] = df_all.loc[rowid]['bytes_done']
    row_dd['bytes_left'] = df_all.loc[rowid]['bytes_left']
    row_dd['current_pos'] = df_all.loc[rowid]['current_pos']
    row_dd['pred_end'] = df_all.loc[rowid]['pred_end']
    return row_dd


#------------------------------------------------------------------------------
# Select (stream_num - 1) call after current call 
#------------------------------------------------------------------------------
def FindComingCalls(df_all_api, r1, stream_num):
    df_all = df_all_api.copy(deep=True)
    lookahead = stream_num - 1
    start_count = False
    count = 0
    result_rows_list = []
    for index, row in df_all.iterrows():
        if index == r1:
            start_count = True 

        if start_count:
            # record current row, avoiding the 1st api 
            if count > 0:  result_rows_list.append(index) 
            if count == lookahead: break
            count = count + 1
            
    return result_rows_list
    

#------------------------------------------------------------------------------
# Get stream id for target row 
#------------------------------------------------------------------------------
def GetStreamID(df_all, r1):
    return df_all.loc[r1]['stream_id']

#------------------------------------------------------------------------------
# Get info for current row 
#------------------------------------------------------------------------------
def GetInfo(df_all, row, column):
    return df_all.loc[row][column]

#------------------------------------------------------------------------------
# Get the time range to check concurrency 
#------------------------------------------------------------------------------
def GetRangeFromWake(df_all_api):
    df_wake = df_all_api.loc[df_all_api.status == 'wake']
    startT = df_wake.current_pos.min()
    endT = df_wake.pred_end.min()
    return startT, endT

#------------------------------------------------------------------------------
# Dump dd 
#------------------------------------------------------------------------------
def Dump_dd(dd):
    for key, value in dd.items():
        print('stream {} : active_api {}'.format(key, value))
