import pandas as pd
import numpy as np
from math import *

from model_param import *

#------------------------------------------------------------------------------
# Figure out when to launch another block for current kernel
#------------------------------------------------------------------------------
def Search_block_start(df_sm_trace, current_kernel_id):
    """
    Read the sm_trace table, find out all the active blocks on current sm, look for the earliest start
    """
    df_active = df_sm_trace.loc[df_sm_trace['active'] == 1]
       
    if not df_active.empty:
        blk2start = df_active['block_start'].max() # find the closest block
        df_active_current_kernel = \
                df_active.loc[df_active['kernel_id'] == current_kernel_id]

        if not df_active_current_kernel.empty:
            # find the closest blk for current kernel
            blk2start = df_active_current_kernel['block_start'].max()
    
        return blk2start
    else:
        # when, on current sm, all the blocks are done/de-activated
        # warning!!!
        return 0.0



def find_sm2start(sm_trace_list, kern_start):
    sm_num = len(sm_trace_list)
    
    AfterPrevKern = False
    
    empSM = 0
    # case 1) there are no trace on each sm
    for df_sm in sm_trace_list:
        if df_sm.empty:
            empSM = empSM + 1 # do nothing

    if empSM == sm_num:
        return 0, AfterPrevKern       
    
    # case 2） there are traces: 
    # by the time where the kernel starts, all the blocks are done already, use sm 0
    max_t = 0
    for df_sm in sm_trace_list:
        cur_max = df_sm.block_end.max()
        if cur_max > max_t:
            max_t = cur_max
            
    if max_t <= kern_start:
        AfterPrevKern = True
        return 0, AfterPrevKern
    else:
        # case 3) : check currently active blocks
        df_sm = sm_trace_list[0]
        df_activeblk = df_sm.loc[df_sm['active'] == 1]
        min_t = df_activeblk.block_end.min()
        target_sm = 0
        
        for i in range(1,sm_num):
            df_sm = sm_trace_list[i]
            df_activeblk = df_sm.loc[df_sm['active'] == 1]
            sm_blk_min = df_activeblk.block_end.min()
            if sm_blk_min < min_t:
                min_t = sm_blk_min
                target_sm = i
                
        return target_sm, AfterPrevKern


#------------------------------------------------------------------------------
# model cke function
#------------------------------------------------------------------------------
def Model_cke_from_same_kernel(Gpu, kernels):
    """
    Model cke using the same kernel
    """
    trace_columns = ['sm_id', 'block_id', 'block_start', 'block_end', 'batch_id', 
            'kernel_id', 'active']

    kernel_num = len(kernels)

    # init SM
    sm_num = Gpu.sm_num
    sms = [SM_Stat() for i in range(sm_num)]
    for i in range(sm_num):
        sms[i].init(Gpu)

    # a trace table to record all the block trace: using pd dataframe
    trace_table = pd.DataFrame(columns=trace_columns)
    sm_trace = [trace_table for x in range(Gpu.sm_num)] # have a trace table for each sm

    #----------------
    # start modeling the trace
    #----------------
    sm2start = 0

    for i in range(kernel_num):
        kern = kernels[i] # schedule current kernel on the device
        kernel_blocks = int(kern.gridDim) # total block for current kern

        last_block_on_sm = 0

        for bid in range(kernel_blocks):
            # find out which sm to allocate
            sm_id = (bid + sm2start) % sm_num

            # check whether current sm has enough resources to host the block
            to_allocate_another_block = check_sm_resource(sms[sm_id], kern)

            #-------------------------------------------
            # There is no more resources to host the blk, consider SM is full now
            # we need to (1) decide how many blks to retire (2) when to start current blk
            #-------------------------------------------
            if to_allocate_another_block == 0:
                # find the list blocks to retire
                df_sm = sm_trace[sm_id]
                df_activeblk = df_sm.loc[df_sm['active'] == 1]

                blkend_min = df_activeblk['block_end'].min()
                df_blk2end = df_activeblk.loc[df_activeblk['block_end'] == blkend_min]
                for index, row in df_blk2end.iterrows():
                    sm_trace[sm_id].loc[index]['active'] = 0 # retire the block
                    sms[sm_id].Rm(kern) # free the block resource

                # after retiring some blocks, we have resources to allocate current block
                sms[sm_id].Allocate_block(kern)

                block_start = blkend_min # when prev blks end, current block starts
                #block_end = block_start + avg_blk_time_list[i] # add avgblktime for currrent kernel
                block_end = block_start + kernels[i].avg_blk_time

                # update the trace table
                sm_trace[sm_id] = sm_trace[sm_id].append({'sm_id': sm_id,
                                                          'block_id': bid,
                                                          'block_start': block_start,
                                                          'block_end' : block_end,
                                                          'batch_id': sms[sm_id].batch,
                                                          'kernel_id': i,
                                                          'active': 1}, ignore_index=True)

            #----------------------------------
            # there is enough resource to host the current block
            #----------------------------------
            if to_allocate_another_block == 1:
                # allocate the block on current sm
                sms[sm_id].Allocate_block(kern)

                # register the block in the trace table
                block_start = None

                # if current sm trace table is empty, start from 0
                # else find the blocks that will end soon, and retire them
                if sm_trace[sm_id].empty:
                    block_start = 0
                else:
                    # read the sm_trace table, find out all the active blocks on current sm, look for the earliest start
                    block_start = Search_block_start(sm_trace[sm_id], i)

                #block_end = block_start + avg_blk_time_list[i]
                block_end = block_start + kernels[i].avg_blk_time

                # add the current block info to the current sm
                sm_trace[sm_id] = sm_trace[sm_id].append({'sm_id': sm_id,
                                                          'block_id': bid,
                                                          'block_start': block_start,
                                                          'block_end' : block_end,
                                                          'batch_id': sms[sm_id].batch,
                                                          'kernel_id': i,
                                                          'active': 1}, ignore_index=True)
            last_block_on_sm = sm_id

        # end of running previous kernel blocks
        sm2start = (last_block_on_sm + 1) % sm_num # start from next smd

    #---------------------------------------------------------------
    # end of for loop to run all the kernel blocks for the cke model
    #---------------------------------------------------------------

    #------------------------
    # for each kernel record the kernel start/end timing
    #------------------------
    kernels_start_end = []

    for kid in range(kernel_num):  # for each kernel, find the corresponding sm trace
        # read sm 0, init the start and end
        df_sm_trace = sm_trace[0]
        df_local_sm = df_sm_trace.loc[df_sm_trace.kernel_id == kid]
        kern_start = df_local_sm.block_start.min()
        kern_end = df_local_sm.block_end.max()

        # read the rest SMs
        for i in range(1, sm_num):
            df_sm_trace = sm_trace[i]
            df_local_sm = df_sm_trace.loc[df_sm_trace.kernel_id == kid]
            #df_local_sm = sm_trace.loc[sm_trace[i].kernel_id == kid] # find the target kernel sm timing
            sm_min = df_local_sm.block_start.min()
            sm_max = df_local_sm.block_end.max()
            if sm_min < kern_start:
                kern_start = sm_min

            if sm_max > kern_end:
                kern_end = sm_max
        # add current kernel start and end time to the list
        kernels_start_end.append([kern_start, kern_end])


    #------------------------------
    # predict time for all the kernels
    #------------------------------
    pred_cke_time = 0.0

    for i in range(sm_num):
        sm_time_max = sm_trace[i]['block_end'].max() - sm_trace[i]['block_start'].min()
        if pred_cke_time < sm_time_max:
            pred_cke_time = sm_time_max

    return pred_cke_time, kernels_start_end
