from math import *
import pandas as pd
import numpy as np
from avgblkmodel import *
from df_util import *

#------------------------------------------------------------------------------
# Find out when to start current stream.
# Read the prevous stream trace, 1) when current h2d exceeds the threshold timing,
# record the current start time, and add the threshold
# 2) if not, the current will start from the last h2d end time for previous one
#------------------------------------------------------------------------------
def find_h2d_start(df_trace, H2D_H2D_OVLP_TH):
    h2d_ovlp = 0
    h2d_starttime = 0
    h2d_endtime = 0

    for index, row in df_trace.iterrows():
        if row.api_type == 'h2d':
            h2d_duation = row['duration']
            h2d_starttime = row['start']  # record the latest h2d
            h2d_endtime = row['end']  # record the latest h2d
            if h2d_duation > H2D_H2D_OVLP_TH:
                h2d_ovlp = 1
                break

        if row.api_type == 'kern': # break when the next is kernel
            break

    stream_start_time = 0.0

    # if there is no overlapping for all h2d api,
    # the second stream will start from the last h2d ending time
    if h2d_ovlp == 0:
        #stream_start_time = h2d_starttime
        stream_start_time = h2d_endtime

    # if there is overlapping, we add the overlapping starting time 
    # with the overlapping threshold
    if h2d_ovlp == 1:
        stream_start_time = h2d_starttime + H2D_H2D_OVLP_TH

    return stream_start_time


#------------------------------------------------------------------------------
# Deep copy the timing trace
# for multiple cuda stream case, each trace will be appended to a list
#------------------------------------------------------------------------------
def init_trace_list(df_trace, stream_num = 1, h2d_ovlp_th = 3.158431):
    df_cke_list = []

    for x in range(stream_num):
        df_dup = df_trace.copy(deep=True)
        df_dup.stream = x # update the stream id
        df_cke_list.append(df_dup)

    #--------------------
    # set up the trace table by adjusting the starting timing
    #--------------------
    for i in range(1,stream_num):
        # compute the time for the previous data transfer
        stream_startTime = find_h2d_start(df_cke_list[i-1], h2d_ovlp_th)
        #print('stream_startTime : {}'.format(stream_startTime))
        df_cke_list[i].start += stream_startTime
        df_cke_list[i].end   += stream_startTime

    return df_cke_list


#------------------------------------------------------------------------------
# Sort api at the 1st time.
# Return the sorted dataframe from the df_cke_list 
#------------------------------------------------------------------------------
def init_sort_api_with_extra_cols(df_cke_list):
    columns_ = ['start', 'end', 'api_type', 'size_kb', 'stream_id', 'status']
    df_all_api = pd.DataFrame(columns=columns_) # init
    stream_num = len(df_cke_list)
    #-------------------------------
    for i in range(stream_num): # read each stream
        stream_id = i
        df_current = df_cke_list[i]
        rows = df_current.shape[0]
        for j in range(rows): # read each row
            start_t = df_current['start'][j]
            end_t = df_current['end'][j]
            api_t = df_current['api_type'][j]
            size_kb = df_current['size'][j]
            df_all_api = df_all_api.append({'start': start_t, 'end': end_t, 
                'api_type': api_t, 'stream_id': stream_id, 'size_kb': size_kb,
                'status': 'sleep'},  ignore_index = True)
    #-------------------------------
    result = df_all_api.sort_values('start', ascending=1) # sort by start col

    # add bandwidth column
    result['bw'] = 0.0
    # compute bandwidth
    for index, row in result.iterrows():
        if row.size_kb > 0.0:
            bw = row.size_kb / (row.end - row.start)
            result.loc[index, 'bw']  = bw


    #-------------------------------
    # add extra columns
    #-------------------------------
    result['bytes_done'] = 0.0
    result['bytes_left'] = result['size_kb']
    result['current_pos'] = 0.0
    result['time_left'] = 0.0
    result['pred_end'] = 0.0

    return result


#------------------------------------------------------------------------------
#  select the first sleep call 
#------------------------------------------------------------------------------
def pick_first_in_sleep(df_all_api):
    df_sleep = df_all_api.loc[df_all_api.status == 'sleep']
    count = 0
    target_rowid = 0
    for index, row in df_sleep.iterrows():
        if count == 0: # 1st row
            target_rowid = index
            break
    target_rowid = int(target_rowid)
    return target_rowid 


#------------------------------------------------------------------------------
# check concurrency 
#------------------------------------------------------------------------------
def check_cc(df_all_api, first, second):
    r1 = first
    r2 = second 

    curapi_start = df_all_api.loc[r1]['start']
    curapi_end = df_all_api.loc[r1]['end']

    nextapi_start = df_all_api.loc[r2]['start']
    nextapi_end = df_all_api.loc[r2]['end']

    #print('{} {} {}'.format(curapi_start, nextapi_start, curapi_end))

    cc = False 
    if curapi_start <= nextapi_start < curapi_end:
        cc = True 
    return cc


#------------------------------------------------------------------------------
# Find the concurrency starting pos and update the  
#------------------------------------------------------------------------------
def update_before_conc(df_all, r1, r2):
    #print('{} {}'.format(r1, r2))
    df_all_api = df_all.copy(deep=True)

    curapi_start = df_all_api.loc[r1]['start']
    curapi_end = df_all_api.loc[r1]['end']
    curapi = df_all_api.loc[r1]['api_type']
    curapi_stream = df_all_api.loc[r1]['stream_id']

    nextapi_start = df_all_api.loc[r2]['start']
    nextapi_end = df_all_api.loc[r2]['end']
    nextapi = df_all_api.loc[r2]['api_type']
    nextapi_stream = df_all_api.loc[r2]['stream_id']

    no_ovlap_time = nextapi_start - curapi_start
    #print('cur start {} next start {}'.format(curapi_start, nextapi_start))
    #print no_ovlap_time

    # the call type for r1 is h2d or d2h
    if curapi in ['h2d', 'd2h'] :
        #print curapi
        curr_trans = df_all_api.loc[r1]['bw'] * no_ovlap_time
        curr_tot   = df_all_api.loc[r1]['size_kb']
        curr_left  = curr_tot - curr_trans
        #print curr_trans 
        #print curr_left

        # update the bytes_done
        df_all_api = UpdateCell(df_all_api, r1, 'bytes_done',  curr_trans)
        df_all_api = UpdateCell(df_all_api, r1, 'bytes_left',  curr_left)
        df_all_api = UpdateCell(df_all_api, r1, 'current_pos', nextapi_start)

    # update the current_pos for r2
    df_all_api = UpdateCell(df_all_api, r2, 'current_pos', nextapi_start)

    return df_all_api


#---------------------------------------------
# model cke function
#---------------------------------------------
def model_cke_from_same_kernel(Gpu, kernels):
    """
    Model cke using the same kernel
    """
    trace_columns = ['sm_id', 'block_id', 'block_start', 'block_end', 'batch_id', 'kernel_id', 'active']

    kernel_num = len(kernels)

    # init SM
    sm_num = Gpu.sm_num
    sms = [sm_stat() for i in range(sm_num)]
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

#---------------------------------------------
# end of model cke function for the same kernel
#---------------------------------------------
