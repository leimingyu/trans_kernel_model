import pandas as pd
import numpy as np
from math import *
import copy # deep copy objects

from model_param import *

#------------------------------------------------------------------------------
# Figure out when to launch another block for current kernel
#------------------------------------------------------------------------------
def Search_block_start(df_sm_trace, current_kernel_id):
    """
    Read the sm_trace table, find out all the active blocks on current sm, 
    look for the earliest start
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


#------------------------------------------------------------------------------
# Figure out which sm to start for current kernel 
#------------------------------------------------------------------------------
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
    
    # case 2) there are traces: by the time where the kernel starts, 
    # all the blocks are done already, use sm 0
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
def cke_model(Gpu, sms_, sm_trace_, kernels_):
    # deep copy the input
    # we need to return the resource and trace for each sm after modeling
    sms = copy.deepcopy(sms_)
    sm_trace = copy.deepcopy(sm_trace_)
    kernels = copy.deepcopy(kernels_)
    
    kernel_num = len(kernels)

    sm_num = Gpu.sm_num
    
    # go through each kernel
    for i in range(kernel_num):
        kern = kernels[i] # schedule current kernel on the device
        kernel_blocks = int(kern.gridDim) # total block for current kern

        kern_start = kern.start_ms

        # 1) find the which sm to start
        # 2) compute whether kernel_start happens before previous kernel ends or not
        sm2start, AfterPrevKern = find_sm2start(sm_trace, kern_start)

        #---------------------------------------------------------
        # Run after previous kernel
        #---------------------------------------------------------
        if AfterPrevKern:
            # deactivate all the previous active blocks
            myid = 0
            for df_sm in sm_trace:
                df_activeblk = df_sm.loc[df_sm['active'] == 1]
                # find the row index of active blocks
                for index, row in df_activeblk.iterrows():     
                    sm_trace[myid].loc[index]['active'] = 0  # deactivate 
                    sms[myid].Rm(kern)    # free the block resource
                    myid = myid + 1



        #---------------------------------------------------------
        # Continue current kernel
        #---------------------------------------------------------
        for bid in range(kernel_blocks):
            sm_id = (bid + sm2start) % sm_num
            to_allocate_another_block = check_sm_resource(sms[sm_id], kern)

            #----------------------------------
            # there is enough resource to host the current block
            #----------------------------------
            if to_allocate_another_block == True:
                # deduct resources on the current sm
                sms[sm_id].Allocate_block(kern)  

                #---------------------------------------
                # register the block in the trace table
                #---------------------------------------
                block_start = None

                offset = 0.0
                # Noted: only the 1st block will adjut the kern_start
                if AfterPrevKern and bid < sm_num: 
                    offset = kern_start

                # if current sm trace table is empty, start from kernel_start
                # else find the blocks that will end soon, and retire them
                if sm_trace[sm_id].empty:
                    block_start = kern_start # (fixed!)
                else:
                    # read the sm_trace table, find out all the active blocks 
                    # on current sm, look for the earliest start
                    block_start = Search_block_start(sm_trace[sm_id], i) + offset

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

            #-------------------------------------------
            # There is no more resources to host the blk, consider SM is full now
            # we need to (1) decide how many blks to retire 
            # (2) when to start current blk
            if to_allocate_another_block == False:
                # find out the active blocks on current sm
                df_sm = sm_trace[sm_id]
                df_activeblk = df_sm.loc[df_sm['active'] == 1]
                df_loc = df_activeblk.copy(deep=True)

                cur_activeblk_num = df_activeblk.shape[0]

                for ii in range(cur_activeblk_num):
                    # find out blocks ending soon
                    blkend_min = df_loc['block_end'].min()
                    df_blk2end = df_loc.loc[df_loc['block_end'] == blkend_min]

                    # retire the blocks
                    for index, row in df_blk2end.iterrows():
                        sm_trace[sm_id].loc[index]['active'] = 0 
                        sms[sm_id].Rm(kern) # free the block resource

                    # enough to allocate a current block
                    if check_sm_resource(sms[sm_id], kern):
                        sms[sm_id].Allocate_block(kern)
                        # when prev blks end, current block starts
                        block_start = blkend_min 
                        # add avgblktime for currrent kernel
                        #block_end = block_start + avg_blk_time_list[i]
                        block_end = block_start + kernels[i].avg_blk_time
                        break # jump out of the loop
                    else:
                        # not enough to allocat another block, remove
                        # Warning: ??? I may just pass
                        #df_loc = df_sm.loc[df_sm['active'] == 1]
                        pass

                # update the trace table
                sm_trace[sm_id] = sm_trace[sm_id].append({'sm_id': sm_id, 
                                                          'block_id': bid, 
                                                          'block_start': block_start,
                                                          'block_end' : block_end,
                                                          'batch_id': sms[sm_id].batch,
                                                          'kernel_id': i,
                                                          'active': 1}, ignore_index=True)
            
        # end of running blocks for current kernel        
            
    #end of kernel iteration
                
    # return the updated sm resource and trace table
    return sms, sm_trace
