import pandas as pd
import numpy as np
from math import *
import sys

from model_param import *


#------------------------------------------------------------------------------
#  select the first sleep call 
#------------------------------------------------------------------------------
def Pick_first_in_sleep(df_all_api):
    df_sleep = df_all_api.loc[df_all_api.status == 'sleep']

    # when apis are 'done' or 'wake', there are no inactive api
    if df_sleep.shape[0] == 0: 
        return None
    else:
        count = 0
        target_rowid = 0
        for index, row in df_sleep.iterrows():
            if count == 0: # 1st row
                target_rowid = index
                break
        target_rowid = int(target_rowid)
        return target_rowid 


#------------------------------------------------------------------------------
#  select the first wake call 
#------------------------------------------------------------------------------
def Pick_first_in_wake(df_all_api):
    df_wake = df_all_api.loc[df_all_api.status == 'wake']
    # when apis are 'done' or 'wake', there are no inactive api
    if df_wake.shape[0] == 0: 
        return None
    else:
        count = 0
        target_rowid = 0
        for index, row in df_wake.iterrows():
            if count == 0: # 1st row
                target_rowid = index
                break
        target_rowid = int(target_rowid)
        return target_rowid 


#------------------------------------------------------------------------------
# Set the target row to be wake status
#------------------------------------------------------------------------------
def AllDone(df_all):
    df_haswork = df_all.loc[df_all.status <> 'done']
    if df_haswork.empty:
        return True
    else:
        return False


#------------------------------------------------------------------------------
#  select two api calls to start prediction 
#------------------------------------------------------------------------------
def PickTwo(df_all_api):
    df_all = df_all_api.copy(deep=True)

    # case 1) : at the beginning, all calls are sleep, select the first two
    df_ActiveAndDone = df_all.loc[df_all.status <> 'sleep']

    all_num = df_all.shape[0]
    wake_num = df_all.loc[df_all.status == 'wake'].shape[0]
    done_num = df_all.loc[df_all.status == 'done'].shape[0]

    sleep_num = df_all.loc[df_all.status == 'sleep'].shape[0]

    # case 1) at the very beginning, or there are only sleep and done (no wake api)
    if df_ActiveAndDone.empty or wake_num == 0:
        # pick the 1st sleep call and wake up
        r1 = Pick_first_in_sleep(df_all)
        #print('r1: {} '.format(r1))
        if r1 is not None:
            df_all = SetWake(df_all, r1) 
        # pick another 
        r2 = Pick_first_in_sleep(df_all)
        if r2 is not None:
            df_all = SetWake(df_all, r2)
    else:
        # case 3) the last api (last sleep)
        if sleep_num == 0 and wake_num == 1: 
            r1 = None
            r2 = None
        else:
            # case 2): during iteration, select the 1st wake, wake up 2nd in sleep
            # there is only sleep one
            r1 = Pick_first_in_wake(df_all)
            r2 = Pick_first_in_sleep(df_all)
            if r2 is not None: df_all = SetWake(df_all, r2)


    print('row:{} row:{}'.format(r1, r2))

    return df_all, r1, r2


#------------------------------------------------------------------------------
# Check whether two api calls are from the same stream 
#------------------------------------------------------------------------------
def Check_stream_id(df_all, r1, r2):
    r1_stream = df_all['stream_id'][r1]
    r2_stream = df_all['stream_id'][r2]

    if r1_stream == r2_stream:
        return True
    else:
        return False


#------------------------------------------------------------------------------
# Check overlapping 
#------------------------------------------------------------------------------
def Check_ovlp(df_all_api, first, second):
    r1 = first
    r2 = second 

    curapi_start = df_all_api.loc[r1]['start']
    curapi_end = df_all_api.loc[r1]['end']

    nextapi_start = df_all_api.loc[r2]['start']
    nextapi_end = df_all_api.loc[r2]['end']

    #print('{} {} {}'.format(curapi_start, nextapi_start, curapi_end))

    ovlp = False 
    if curapi_start <= nextapi_start < curapi_end:
        ovlp = True 
    return ovlp 


#------------------------------------------------------------------------------
# Find the concurrency starting pos and update the  
#------------------------------------------------------------------------------
def Update_before_ovlp(df_all, r1, r2):
    df_all_api = df_all.copy(deep=True)
    #print('{} {}'.format(r1, r2))

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

    #----------------------------
    # update r1 with current pos
    #----------------------------
    df_all_api = UpdateCell(df_all_api, r1, 'current_pos', nextapi_start)

    # the call type for r1 is h2d or d2h
    if curapi in ['h2d', 'd2h'] :
        curr_trans = df_all_api.loc[r1]['bw'] * no_ovlap_time # full bw since no ovlp
        curr_tot   = df_all_api.loc[r1]['size_kb']
        curr_left  = curr_tot - curr_trans
        # update the bytes_done
        df_all_api = UpdateCell(df_all_api, r1, 'bytes_done',  curr_trans)
        df_all_api = UpdateCell(df_all_api, r1, 'bytes_left',  curr_left)

    #----------------------------
    # update r2 with current pos
    #----------------------------
    df_all_api = UpdateCell(df_all_api, r2, 'current_pos', nextapi_start)

    return df_all_api


#------------------------------------------------------------------------------
# Predict the end time when there is no conflict. 
#------------------------------------------------------------------------------
def Predict_noConflict(df_all, first, second):
    df_all_api = df_all.copy(deep=True)

    target_rows = [first, second]

    for r1 in target_rows:  # work on the target row 
        r1_type = df_all_api.loc[r1]['api_type']
        cur_pos = df_all_api.loc[r1]['current_pos']

        # update the predicted end time based on the api type
        if r1_type in ['h2d', 'd2h']:
            # check the bytes left and use bw to predict the end time
            bw = df_all_api.loc[r1]['bw']
            bytesleft = df_all_api.loc[r1]['bytes_left']
            pred_time_left = bytesleft / bw
            df_all_api = UpdateCell(df_all_api, r1, 'pred_end', cur_pos + pred_time_left)
        elif r1_type == 'kern':
            # no overlapping, no change to kernel time: curpos + kernel_runtime
            kernel_time = df_all_api.loc[r1]['end'] - df_all_api.loc[r1]['start']
            df_all_api = UpdateCell(df_all_api, r1, 'pred_end', kernel_time + cur_pos)
        else:
            sys.stderr.write('Unknown API call.')

    return df_all_api 


#------------------------------------------------------------------------------
# Predict the end time when there concurrency for data transfer 
#------------------------------------------------------------------------------
def Predict_transCC(df_all, first, second):
    df_all_api = df_all.copy(deep=True)

    cc = 2.0
    
    row_list = [first, second]

    for i in row_list:
        # the bandwidth is shared among all the concurrent transfer
        bw = df_all_api.loc[i]['bw'] / cc
        # predict the transfer time based on the bandwidth
        cur_pred_time_left = df_all_api.loc[i]['bytes_left'] / bw
        # update the future ending time
        df_all_api = UpdateCell(df_all_api, i, 'pred_end', 
                cur_pred_time_left + df_all_api.loc[i]['current_pos'] )

    return df_all_api 


#------------------------------------------------------------------------------
# Predict the ending time: based on the concurrency
# 1) if they are both h2d_h2d, d2h_d2h or kern_kern, we need to predict use different mode
# 2) if they are different apis, there is no interference
#------------------------------------------------------------------------------
def Predict_end(df_all, r1, r2, ways = 1.0):
    """
    From the input dataframe, adjust the row info, depending on the concurrency.
    """
    df_all_api = df_all.copy(deep=True)

    cc = ways # concurrency

    r1_apitype = df_all_api.loc[r1]['api_type']
    r2_apitype = df_all_api.loc[r2]['api_type']

    interference = True if r1_apitype == r2_apitype else False

    if interference == False:
        df_all_api = Predict_noConflict(df_all_api, r1, r2)
    else:
        if r1_apitype in ['h2d', 'd2h']: # data transfer model
            df_all_api = Predict_transCC(df_all_api, r1, r2)
        elif r1_apitype == 'kern': # todo: cke model
            pass 
        else:
            sys.stderr.write('Unknown API call.')
    
    return df_all_api


#------------------------------------------------------------------------------
# get the time range from wake api, to check the next concurrent api 
#------------------------------------------------------------------------------
def Get_pred_range(df_all):
    df_wake = df_all.loc[df_all.status == 'wake']
    begT = df_wake.current_pos.min()
    endT = df_wake.pred_end.min()
    return [begT, endT]


#------------------------------------------------------------------------------
# check concurrency by another cuda stream within a time range 
#------------------------------------------------------------------------------
def Check_cc_by_time(df_all, time_range):
    df_all_api = df_all.copy(deep=True)
    df_wake = df_all_api.loc[df_all_api.status == 'wake']
    df_sleep = df_all_api.loc[df_all_api.status == 'sleep']

    # find out the stream ids in df_wake
    new_stream_ls = []
    for x in df_sleep.stream_id.unique():
        if x not in df_wake.stream_id.unique():
            new_stream_ls.append(x)
    
    has_conc_stream = 1 if new_stream_ls else 0;
    #print('has_conc_stream {}'.format(has_conc_stream))

    # todo:
    # look for streams that start within the time range
    extra_cc = 0
    if has_conc_stream == 1:
        for sid in new_stream_ls:
            df_cur = df_sleep.loc[df_sleep.stream_id == sid]
            for index, row in df_cur.iterrows():
                startT = row.start
                if time_range[0] <= startT < time_range[1]: # api in the range
                    extra_cc = 1

    return extra_cc 


#------------------------------------------------------------------------------
# Update the ending time: based on the concurrency
#------------------------------------------------------------------------------
def Update_ovlpTrans(df_all, timerange_list, ways = 1.0):
    startT = timerange_list[0]
    endT = timerange_list[1]
    dur = endT - startT

    cc = ways

    df_all_api = df_all.copy(deep=True)

    # since the df_all_api are sorted by start
    # we only need to check the wake stream and start from top
    for index, row in df_all_api.iterrows():
        if row.status == 'wake':
            bw = row.bw / cc
            bytes_don = row.bytes_done
            bytes_lft = row.bytes_left
            bytes_tran = dur * bw 
            bytes_left = row.bytes_left - bytes_tran

            done = 0
            if abs(bytes_left - 0.0) <  1e-3: #  smaller than 1 byte
                done = 1

            #print index

            if done == 1:
                # update bytes_done
                tot_size = row.size_kb
                #print tot_size
                df_all_api.set_value(index,'bytes_done', tot_size)
                df_all_api.set_value(index,'bytes_left', 0)
                df_all_api.set_value(index,'time_left', 0) # no time_left
                df_all_api.set_value(index,'current_pos', row.pred_end)
                df_all_api.set_value(index,'status', 'done')
            else:
                # deduct the bytes, update teh current pos
                df_all_api.set_value(index,'bytes_done', bytes_don + bytes_tran)
                df_all_api.set_value(index,'bytes_left', bytes_lft - bytes_tran)
                df_all_api.set_value(index,'current_pos', endT)
                df_all_api.set_value(index,'time_left', 0) # clear
                df_all_api.set_value(index,'pred_end', 0) # clear

    return df_all_api


#------------------------------------------------------------------------------
# For cuda calls with 'done' status, update the timing for that stream 
#------------------------------------------------------------------------------
def UpdateStreamTime(df_all_api):
    # copy the input
    df_all = df_all_api.copy(deep=True)
    df_done = df_all.loc[df_all.status == 'done'] # find out which api is done

    if df_done.empty:
        return df_all

    done_streams = df_done.stream_id.unique() # np.array

    for x in done_streams:
        # read the stream
        df_cur = df_all.loc[df_all.stream_id == x] # the api in order

        prev_start = 0.0
        prev_end = 0.0
        prev_pred_end = 0.0
        prev_status = '' 
        prev_newEnd = 0.0

        count = 0
        for index, row in df_cur.iterrows(): # process each row
            # record previous timing and status
            if count == 0:
                prev_start = row.start
                prev_end = row.end
                #print('prev_end {}'.format(prev_end))
                prev_pred_end = row.pred_end
                prev_status = row.status

            # read current stat
            cur_start = row.start 
            #print('cur_start {}'.format(cur_start))
            cur_end = row.end
            cur_pred_end = row.pred_end
            cur_status = row.status

            #print('count {} : cur_start {}  prev_end {}'.format(count, cur_start, prev_end)) 

            if cur_status == 'done':
                # if it is done, no need to update, save it for coming row
                prev_start = row.start
                prev_end = row.end
                prev_pred_end = row.pred_end
                prev_status = row.status
            else:
                # adjust offset according to the previous predicted_end
                ovhd = cur_start - prev_end 
                #print('stream {} : cur_start {}'.format(x, cur_start))

                if prev_status == 'done':
                    new_start = prev_pred_end + ovhd    # offset with the pred_end
                else:
                    new_start = prev_newEnd + ovhd  # with previous new_end

                new_end = new_start + (cur_end - cur_start)  # new start + duration

                # before updating the current record, save the current 
                prev_start = cur_start
                prev_end = cur_end
                prev_pred_end = cur_pred_end
                prev_status = cur_status
                prev_newEnd = new_end # important!

                # update the dataframe record
                #print index
                df_all.set_value(index, 'start',    new_start)
                df_all.set_value(index, 'end',      new_end)
                df_all.set_value(index, 'pred_end', new_end)


            # update the count for current iter
            count = count + 1

        # update the end column for rows with 'done' status
        df_cur_done = df_cur.loc[df_cur.status == 'done']
        for index, row in df_cur_done.iterrows():
            df_all.set_value(index, 'end', row.pred_end) # update with pred_end

        #----------------------
        # end of current stream

    #--------------------------------------
    #end of all the streams with 'done' call

    return df_all


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

    ## warning : add api launch ovhd
    stream_start_time += 0.002

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
        print('stream_startTime : {}'.format(stream_startTime))

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
    # generate the trace table 
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
    # sort by the start column
    #-------------------------------
    result = df_all_api.sort_values('start', ascending=1)

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
    #result['time_left'] = 0.0
    result['pred_end'] = 0.0

    return result


#------------------------------------------------------------------------------
#  check concurrency during an interval for wake api calls
#------------------------------------------------------------------------------
def Check_CC(df_wake, begT, endT):
    cc = 0.0
    cc_rows = []
    for index, row in df_wake.iterrows():
        mystart = row.start
        myend   = row.end
        
        if mystart < endT : # if current wake starts before the end time, add cc
            cc = cc + 1.0
            cc_rows.append(index) # find out the row index
    return cc, cc_rows



#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def Update_row_by_cc(df_all, r, cc, timeRange):
    df = df_all.copy(deep=True)

    begT = timeRange[0]
    endT = timeRange[1]
    duration =  endT - begT

    my_type   = df.loc[r]['api_type']
    my_curpos = df.loc[r]['current_pos']
    my_end    = df.loc[r]['end']
    my_pred_end    = df.loc[r]['pred_end']
    my_kb     = df.loc[r]['size_kb']

    my_left_new = 0.0
    my_bytes_done_new = 0.0
    my_left_time = 0.0

    if my_type in ['h2d', 'd2h']:
        my_bytes_left = df.loc[r]['bytes_left']
        my_bytes_done = df.loc[r]['bytes_done']
        my_bw         = df.loc[r]['bw'] / cc

        bytes_tran = duration * my_bw

        my_left_time = my_bytes_left / my_bw

        if my_bytes_left < 1e-3:
            sys.stderr.write('no bytes left')
        
        # calculate how many bytes left
        my_left_new = my_bytes_left - bytes_tran
        print('row {}, end {}, my_curpos {}, my_pred_end {}, my_left_new {}'.format(r, my_end, my_curpos, my_pred_end, my_left_new))

        if my_left_new < 1e-3:
            my_left_new = 0.0

        # compute bytes done so far
        my_bytes_done_new = my_bytes_done + bytes_tran

    df = UpdateCell(df, r, 'current_pos', endT)

    if my_type in ['h2d', 'd2h']:
        df = UpdateCell(df, r, 'bytes_left', my_left_new)
        df = UpdateCell(df, r, 'bytes_done', my_bytes_done_new)
        if my_left_new == 0.0:
            my_new_end =  my_curpos + my_left_time
            #print('my_new_end {}'.format(my_new_end))
            # WARNING: use the org end time : 
            # if current call is done by the time minT starts
            df= UpdateCell(df, r, 'current_pos', my_new_end) 
            df= UpdateCell(df, r, 'pred_end', my_new_end)  # update
            df= UpdateCell(df, r, 'bytes_done', my_kb)
            df= UpdateCell(df, r, 'status', 'done')

    return df


#------------------------------------------------------------------------------
# Predict new end time based on the concurrency 
#------------------------------------------------------------------------------
def Adjust_pred(df_all, cc, cc_row_list):
    df = df_all.copy(deep=True)
    for rid in cc_row_list:
        my_type = GetInfo(df, rid, 'api_type') 
        my_curpos = GetInfo(df, rid, 'current_pos') 
        if my_type in ['h2d', 'd2h']:
            my_bw = GetInfo(df, rid, 'bw') 
            my_bw = my_bw / cc

            my_bytes_left = GetInfo(df, rid, 'bytes_left')
            trans_time = my_bytes_left / my_bw
            # new predicted end
            my_new_end = my_curpos + trans_time
            # update 
            df= UpdateCell(df, rid, 'pred_end', my_new_end)

    return df


#------------------------------------------------------------------------------
# Start the target api, check prev ovlapping, and update the timing accordingly 
#
# WARNING: 
# 1) if one call is done, the prediction using cc still happen on other call
#    where we probably over-predict the ending time 
#------------------------------------------------------------------------------
def MoveCurPos(df_all, r1):
    df = df_all.copy(deep=True)

    df_wake = df.loc[df.status == 'wake']
    #print df_wake

    wake_api_num = df_wake.shape[0]
    #print wake_api_num

    # get the range to check ovlp
    begT = df_wake.current_pos.min()
    endT = df_wake.pred_end.min()

    midT = endT

    # check any wake api start between the range
    for index, row in df_wake.iterrows():
        cur_start = row.start
        if  begT < cur_start < endT:
            midT = cur_start

    #print midT

    # check concurrency [begT,midT]
    cc, cc_rows = Check_CC(df_wake, begT, midT)
    print('from {} to {}, cc = {}'.format(begT, midT, cc))
    print cc_rows
    # predict based on current cc
    for r in cc_rows:
        df = Update_row_by_cc(df, r, cc, [begT, midT])

    if midT < endT:
        cc_new, cc_rows_new = Check_CC(df_wake, midT, endT)
        print('from {} to {}, cc = {}'.format(midT, endT, cc_new))
        print cc_rows_new

        # when the concurrency changes: update the pred_end based on current cc
        if cc_new <> cc:
            df = Adjust_pred(df, cc_new, cc_rows_new)

        # find out the new time range to predict
        chk_start, chk_end = GetRangeFromWake(df)
        print('new pred range from {} to {}'.format(chk_start, chk_end))

        # update row during this range
        for r in cc_rows:
            df = Update_row_by_cc(df, r, cc_new, [chk_start, chk_end])


    # check whether any api call has ended
    # if so, update timing for all the calls in the stream 
    df = UpdateStreamTime(df)

    return df


#------------------------------------------------------------------------------
# start next api 
# 1) if in sleep, wake it up  2) if active, directly return
#------------------------------------------------------------------------------
def start_next_call(df_all, prev_row):
    df = df_all.copy(deep=True)

    row_id = None
    row_stream = None

    #df_sleep = df.loc[df.status == 'sleep']

    #if df_sleep.shape[0] > 0:
    #    # pick the 1st one in sleep
    #    row_id     = Pick_first_in_sleep(df)
    #    df         = SetWake(df, row_id)
    #    row_stream = GetInfo(df, row_id, 'stream_id')


    df_nodone = df.loc[df.status <> 'done']

    found_prev = False
    for index, row in df_nodone.iterrows():
        if found_prev:
            row_id = index # the row after prev_row
            break

        if index == prev_row:
            found_prev = True


    my_status = GetInfo(df, row_id, 'status')

    if my_status == 'sleep':
        df         = SetWake(df, row_id)
        row_stream = GetInfo(df, row_id, 'stream_id')

    if my_status == 'wake':
        row_stream = GetInfo(df, row_id, 'stream_id')

    return df, row_id, row_stream

#------------------------------------------------------------------------------
# start next api 
# todo: add cases for kernels
#------------------------------------------------------------------------------
def StartNext_byType(df_all, row_list):
    df_all_api = df_all.copy(deep=True)

    # row r1 and r2 should be wake
    r1 = row_list[0]
    r2 = row_list[1]

    #----------------
    # row r2
    #----------------
    r2_start = df_all_api.loc[r2]['start']

    #----------------
    # row r1: previous one that in wake
    #----------------
    r1_type = df_all_api.loc[r1]['api_type']
    r1_cur_pos = df_all_api.loc[r1]['current_pos']
    r1_end = df_all_api.loc[r1]['end']
    r1_left_new = 0.0
    r1_bytesdone_new = 0.0
    r1_kb = 0.0


    # if r1 type is transfer call, we need to update the transfer status
    if r1_type in ['h2d', 'd2h']:
        r1_bw = df_all_api.loc[r1]['bw']
        r1_bytesdone = df_all_api.loc[r1]['bytes_done']
        r1_kb = df_all_api.loc[r1]['size_kb']
        # compute trans size
        duration = r2_start - r1_cur_pos
        r1_bytes_tran = duration * r1_bw

        # check bytes left
        r1_bytes_left = df_all_api.loc[r1]['bytes_left']
        #print('bytes left : {}'.format(r1_bytes_left))
        if r1_bytes_left < 1e-3:
            sys.stderr.write('no bytes left')

        # calculate how many bytes left
        r1_left_new = r1_bytes_left - r1_bytes_tran
        if r1_left_new < 1e-3:
            r1_left_new = 0.0

        # compute bytes done so far
        r1_bytesdone_new = r1_bytesdone + r1_bytes_tran

    # update r1 status
    if r2_start < r1_end: # r2 starts before r1 ends
        df_all_api = UpdateCell(df_all_api, r1, 'current_pos', r2_start)
    else:   # r2 start after r1 ends
        df_all_api = UpdateCell(df_all_api, r1, 'current_pos', r1_end)

    if r1_type in ['h2d', 'd2h']:
        df_all_api = UpdateCell(df_all_api, r1, 'bytes_left', r1_left_new)
        df_all_api = UpdateCell(df_all_api, r1, 'bytes_done', r1_bytesdone_new)
        if r1_left_new == 0.0:
            # WARNING: use the org end time : 
            # if current call is done by the time r2 starts
            df_all_api = UpdateCell(df_all_api, r1, 'current_pos', r1_end) 
            df_all_api = UpdateCell(df_all_api, r1, 'pred_end', r1_end)  # update
            df_all_api = UpdateCell(df_all_api, r1, 'bytes_done', r1_kb)
            df_all_api = UpdateCell(df_all_api, r1, 'status', 'done')

    # update r2 status: current pos
    df_all_api = UpdateCell(df_all_api, r2, 'current_pos', r2_start)

    #print('r1 : {}, r2 : {}, r2_start : {}'.format(r1, r2, r2_start))

    return df_all_api 


#------------------------------------------------------------------------------
# update time between an interval  
#------------------------------------------------------------------------------
def update_by_range(df_all, begT, endT):
    df = df_all.copy(deep=True)

    # find out the wake api during the range
    wake_list = GetWakeListByTime(df, begT, endT)
    print('wake list {} '.format(wake_list))

    # no wake api
    if not wake_list:
        return df
    
    # how many h2d ovlp
    h2d_list, d2h_list, kern_list = FindOvlp(df, wake_list)
    print h2d_list
    print d2h_list
    print kern_list

    # check whether there is h2d ovlp
    if h2d_list:
        cc = len(h2d_list)
        print('cc {} for all the wake list'.format(cc))
        for r in h2d_list: 
            df = Update_h2d_bytes(df, r, begT, endT, ways = cc)
        # check any h2d call

    # check whether there is d2h ovlp
    if d2h_list:
        pass

    # check whether there is kern ovlp
    if kern_list:
        pass

    #for rowid in wake_list:


    return df

#------------------------------------------------------------------------------
# check active stream dd and terminate an api that ends soon
#------------------------------------------------------------------------------
def check_activestream_and_update(df_all, activestream_dd, simPos):
    df = df_all.copy(deep=True)

    full = True
    for key, value in activestream_dd.items():
        if value == None:
            full = False 
            break

    if not full:
        return df

    # find out which call to terminate
    df_wake = df.loc[df.status == 'wake']
    wake_list = FindWakeList(df_wake) 
    print(wake_list)

    # sort
    df_sorted = df_wake.sort_values(['pred_end'],ascending = True)
    #print df_sorted

    row2end = df_sorted.iloc[0].name
    row2end_stream = df_sorted.iloc[0]['stream_id']
    print('row to end : {}, its stream {}'.format(row2end, row2end_stream))
    #print type(row2end)

    #
    # find out the next call after row2nd
    row_afterprevcall = Find_nextcall_samestream(df, row2end, row2end_stream)
    nextCall_start = GetInfo(df, row_afterprevcall, 'start')

    #
    # end the target row, update the bytes for other call
    df = end_target_row(df, row2end, simPos, nextCall_start)

    # 
    # move current_pos to row2nd pred_end
    row2end_predend = GetInfo(df, row2end, 'pred_end')
    for wake_row in wake_list:
        if wake_row <> row2end:
            df = UpdateCell(df, wake_row, 'current_pos', row2end_predend)

    #
    # move to row2end_end to nextcall_start
    df = move_wake_for_coming_call(df, row2nd_predend, nextCall_start)




    return df

#------------------------------------------------------------------------------
# Move wake calls to the coming api start: no ovlp during the rangeT
# Warning: there are cases where overlapping exists
#------------------------------------------------------------------------------
def move_wake_for_coming_call(df_all, preEndT, curStartT):
    df = df_all.copy(deep=True)
    wake_list = GetWakeListBefore(df, preEndT)
    print('move_wake_for_coming_call, wake list {} '.format(wake_list))
    #
    dur = curStartT - preEndT

    for wake_row in wake_list:
        wake_row_api = GetInfo(df, wake_row, 'api_type')
        if wake_row_api in ['h2d', 'd2h']:
            bw = GetInfo(df, wake_row, 'bw')
            bytes_left = GetInfo(df, wake_row, 'bytes_left')
            bytes_done = GetInfo(df, wake_row, 'bytes_done')
            trans_bytes = dur * bw
            bytes_left_new = bytes_left - trans_bytes
            bytes_done_new = bytes_done + trans_bytes
            #
            # no need to update pred_end, since Update_row_h2d assume on ovlp
            df = UpdateCell(df, wake_row, 'bytes_left',  bytes_left_new)
            df = UpdateCell(df, wake_row, 'bytes_done',  bytes_done_new)
            df = UpdateCell(df, wake_row, 'current_pos', curStartT)
    return df

#------------------------------------------------------------------------------
# finish the target row and update the timing 
#------------------------------------------------------------------------------
def end_target_row(df_all, row2nd, simT, curT):
    df = df_all.copy(deep=True)
    #
    # find wake apis before curT
    wake_list = GetWakeListBefore(df, curT)
    print('wake list {} '.format(wake_list))
    #
    # check row2nd api type
    mytype = GetInfo(df, row2nd, 'api_type')

    if mytype == 'h2d':
        #
        # how many h2d ovlp during the interval
        h2d_list, _, _ = FindOvlp(df, wake_list)
        cc = len(h2d_list)
        print cc
        #
        # finish current row and update the pred time
        df = Finish_row_h2d(df, row2nd, simT, ways = cc)
        #
        # if an api is done, update the timing for the stream 
        df = UpdateStreamTime(df)

        pred_end = GetInfo(df, row2nd, 'pred_end') 

        if cc > 1.0:
            # update the time for other stream
            for x in h2d_list:
                if x <> row2nd:
                    # update bytes_left and bytes_done
                    df = Update_row_h2d(df, x, simT, pred_end, ways = cc)


    if mytype == 'd2h':
        sys.stderr.write('end_target_row, d2h not implemented')
        pass


    if mytype == 'kern':
        sys.stderr.write('end_target_row, kern not implemented')
        pass



    return df


#------------------------------------------------------------------------------
# check concurrency using current_pos
#------------------------------------------------------------------------------
def Predict_checkCC(df_all, first, second):
    df_all_api = df_all.copy(deep=True)
    r1 = first
    r2 = second 
    
    # if r1 current_pos == r2 start, there is overlapping
    r1_cur_pos = df_all_api.loc[r1]['current_pos']
    r2_start = df_all_api.loc[r2]['start']

    conc = 0
    if r1_cur_pos == r2_start:  # when the two api start at the same time
        conc = 1

    # when there is overlapping
    if conc == 1:
        cc = 2.0
        # predcit the next 
        df_all_api = Predict_end(df_all_api, r1, r2, ways = cc)

    return df_all_api 


#------------------------------------------------------------------------------
# Check the api type or not, return type
#------------------------------------------------------------------------------
def CheckType(df_all, r1, r2):
    r1_type = df_all.loc[r1]['api_type']
    r2_type = df_all.loc[r2]['api_type']

    whichType = None 
    if r1_type == r2_type:
        whichType = r1_type 

    return whichType



#------------------------------------------------------------------------------
# Update using pred_end when there is no conflict. 
#------------------------------------------------------------------------------
def Update_wake_noConflict(df_all, timeRange):
    df_all_api = df_all.copy(deep=True)
    df_wake = df_all_api.loc[df_all_api.status == 'wake'] # wake apis

    startT = timeRange[0]
    endT = timeRange[1]
    dur = endT - startT

    # iterate through each row to update the pred_end
    for index, row in df_wake.iterrows():
        apitype = row.api_type
        if apitype in ['h2d', 'd2h']: # for transfer, we need to update the bytes also
            bw = row.bw
            bytes_tran = dur * bw 
            bytes_don = row.bytes_done
            bytes_lft = row.bytes_left
            bytes_left = row.bytes_left - bytes_tran

            done = 0
            if abs(bytes_left - 0.0) <  1e-3: #  smaller than 1 byte
                done = 1

            if done == 1:
                # update bytes_done
                tot_size = row.size_kb
                #print tot_size
                df_all_api.set_value(index,'bytes_done', tot_size)
                df_all_api.set_value(index,'bytes_left', 0)
                df_all_api.set_value(index,'time_left', 0) # no time_left
                df_all_api.set_value(index,'current_pos', row.pred_end)
                df_all_api.set_value(index,'status', 'done')
            else:
                # deduct the bytes, update teh current pos
                df_all_api.set_value(index,'bytes_done', bytes_don + bytes_tran)
                df_all_api.set_value(index,'bytes_left', bytes_lft - bytes_tran)
                df_all_api.set_value(index,'current_pos', endT)
                df_all_api.set_value(index,'time_left', 0) # clear
                df_all_api.set_value(index,'pred_end', 0) # clear

        elif apitype == 'kern': # update current_pos and status
            k_pred_end = row.pred_end
            k_start = row.start
            k_end = row.end
            if k_pred_end > k_end: # there is more work to do
                df_all_api.set_value(index, 'current_pos', endT)
                df_all_api.set_value(index, 'time_left', k_pred_end - k_end)
            else: # the kernel is done
                df_all_api.set_value(index, 'current_pos', k_pred_end)
                df_all_api.set_value(index, 'status', 'done')

        else:
            sys.stderr.write('Unknown API call.')

    return df_all_api 


#------------------------------------------------------------------------------
# Predict the end time when there is memory transfer overlapping. 
#------------------------------------------------------------------------------
def Predict_transferOvlp(df_all, first, second, ways = 1.0):
    df_all_api = df_all.copy(deep=True)

    target_rows = [first, second]

    cc = ways

    for r1 in target_rows:  # work on the target row 
        r1_type = df_all_api.loc[r1]['api_type']
        cur_pos = df_all_api.loc[r1]['current_pos']

        # check the bytes left and use bw to predict the end time
        bw = df_all_api.loc[r1]['bw'] / cc
        bytesleft = df_all_api.loc[r1]['bytes_left']
        pred_time_left = bytesleft / bw
        df_all_api = UpdateCell(df_all_api, r1, 'pred_end', cur_pos + pred_time_left)

    return df_all_api 


def Predict_transferOvlp(df_all, row_list):
    df_all_api = df_all.copy(deep=True)

    cc = float(len(row_list))

    for r1 in row_list:  # work on the target row 
        cur_pos = df_all_api.loc[r1]['current_pos']

        # check the bytes left and use bw to predict the end time
        bw = df_all_api.loc[r1]['bw'] / cc
        bytesleft = df_all_api.loc[r1]['bytes_left']
        pred_time_left = bytesleft / bw
        df_all_api = UpdateCell(df_all_api, r1, 'pred_end', cur_pos + pred_time_left)

    return df_all_api 


#------------------------------------------------------------------------------
# Update using pred_end when there is no conflict. 
#------------------------------------------------------------------------------
def Update_wake_transferOvlp(df_all, timeRange, ways = 1.0):
    df_all_api = df_all.copy(deep=True)
    df_wake = df_all_api.loc[df_all_api.status == 'wake'] # wake apis

    startT = timeRange[0]
    endT = timeRange[1]
    dur = endT - startT

    cc = ways

    # iterate through each row to update the pred_end
    for index, row in df_wake.iterrows():
        bw = row.bw / cc
        bytes_tran = dur * bw 
        bytes_don = row.bytes_done
        bytes_lft = row.bytes_left
        bytes_left = row.bytes_left - bytes_tran

        done = 0
        if abs(bytes_left - 0.0) <  1e-3: #  smaller than 1 byte
            done = 1

        if done == 1:
            # update bytes_done
            tot_size = row.size_kb
            #print tot_size
            df_all_api.set_value(index,'bytes_done', tot_size)
            df_all_api.set_value(index,'bytes_left', 0)
            df_all_api.set_value(index,'time_left', 0) # no time_left
            df_all_api.set_value(index,'current_pos', row.pred_end)
            df_all_api.set_value(index,'status', 'done')
        else:
            # deduct the bytes, update teh current pos
            df_all_api.set_value(index,'bytes_done', bytes_don + bytes_tran)
            df_all_api.set_value(index,'bytes_left', bytes_lft - bytes_tran)
            df_all_api.set_value(index,'current_pos', endT)
            df_all_api.set_value(index,'time_left', 0) # clear
            df_all_api.set_value(index,'pred_end', 0) # clear

    return df_all_api 


#------------------------------------------------------------------------------
# consider the two kernels are done, update the current pos 
#------------------------------------------------------------------------------
def Update_wake_kernOvlp(df_all):
    df_all_api = df_all.copy(deep=True)
    df_wake = df_all_api.loc[df_all_api.status == 'wake'] # wake apis
    df_wake_kern = df_wake.loc[df_wake.api_type== 'kern'] # wake kernels 

    # iterate through each row to update the pred_end
    for index, row in df_wake_kern.iterrows():
        df_all_api.set_value(index,'current_pos', row.pred_end)
        df_all_api.set_value(index,'status', 'done')

    return df_all_api 


#------------------------------------------------------------------------------
# For the last api call, update the entire trace table. 
#------------------------------------------------------------------------------
def UpdateStream_lastapi(df_all_api):
    # copy the input
    df_all = df_all_api.copy(deep=True)
    df_lastwake = df_all.loc[df_all.status == 'wake'] # find out the last active api 

    for index, row in df_lastwake.iterrows():
        apitype = row.api_type
        if apitype in ['h2d', 'd2h']: # there is no overlapping since the last one
            bw = row.bw
            cur_pos = row.current_pos
            bytes_left = row.bytes_left # bytes left to transfer
            time_to_finish= bytes_left / bw
            pred_end = cur_pos + time_to_finish
            # compute the new end : cur_pos + time_to_finish 
            df_all.set_value(index, 'pred_end', pred_end)
            df_all.set_value(index, 'bytes_left', 0)
            df_all.set_value(index, 'bytes_done', row.size_kb)
            #df_all.set_value(index, 'time_left', 0)
            df_all.set_value(index, 'status', 'done')
            df_all.set_value(index, 'current_pos', pred_end) # current will be the pred_end
            df_all.set_value(index, 'end', pred_end) # end will be the pred_end

    return df_all


#------------------------------------------------------------------------------
# Check whether any row is done
#------------------------------------------------------------------------------
def CheckRowDone(df_all, r1, r2):
    r1_status = df_all.loc[r1]['status']
    r2_status = df_all.loc[r2]['status']

    next_iter = False
    if r1_status == 'done' or r2_status == 'done':
        next_iter = True

    return next_iter


def CheckRowDone(df_all, row_list):
    next_iter = False
    for r1 in row_list:
        r1_status = df_all.loc[r1]['status']
        if r1_status == 'done':
            next_iter = True
            break

    done_list = []
    for r1 in row_list:
        r1_status = df_all.loc[r1]['status']
        if r1_status == 'done':
           done_list.append(r1) 

    return next_iter, done_list


#------------------------------------------------------------------------------
# Check whether any row is done
#------------------------------------------------------------------------------
def FindStreamAndKernID(df_all_api, r1):
    stream_id = df_all_api.loc[r1]['stream_id']
    stream_id = int(stream_id)

    df_stream = df_all_api.loc[df_all_api.stream_id == stream_id]
    # iterate through each row, count when the index == r1
    kernel_id = 0
    kcount = 0
    for index, row in df_stream.iterrows():
        if row.api_type == 'kern':
            kcount = kcount + 1
            if index == r1:
                kernel_id = kcount - 1  # index kernel from 0 

    kernel_id = int(kernel_id)
    return stream_id, kernel_id


#------------------------------------------------------------------------------
# Get the start time for the current row/api call. 
#------------------------------------------------------------------------------
def GetStartTime(df_all_api, r1):
    return float(df_all_api.loc[r1]['start'])


#------------------------------------------------------------------------------
#  
#------------------------------------------------------------------------------
def pick_first_call(df_all_api, mode = 'sleep'):
    df_all = df_all_api.copy(deep=True)

    df_sleep = df_all.loc[df_all.status == mode]

    count = 0
    target_rowid = 0
    target_stream = 0
    for index, row in df_sleep.iterrows():
        if count == 0: # 1st row
            target_rowid = index
            target_stream = row.stream_id
            break

    return df_all, int(target_rowid), target_stream 


#------------------------------------------------------------------------------
# pick an api to start 
#------------------------------------------------------------------------------
def pick_base_call(df_all):
    df = df_all.copy(deep=True)
    total_calls = df.shape[0]

    df_sleep = df.loc[df.status == 'sleep']
    df_wake  = df.loc[df.status == 'wake']
    sleep_num = df_sleep.shape[0]

    r1 , r1_stream = None, None

    if sleep_num == total_calls: # all sleep
        df, r1, r1_stream = pick_first_call(df, mode = 'sleep')

    if not df_wake.empty:
        df, r1, r1_stream = pick_first_call(df, mode = 'wake')

    return df, r1, r1_stream

#------------------------------------------------------------------------------
# find unique streams in the dataframe 
#------------------------------------------------------------------------------
def find_unique_streams(df_all_api):
    df_all = df_all_api.copy(deep=True)
    results = list(df_all.stream_id.unique()) # numpy array to list
    return results


#------------------------------------------------------------------------------
# Finish current call 
#------------------------------------------------------------------------------
def finish_call(df_all, row):
    df = df_all.copy(deep=True)
    my_type = GetInfo(df, row, 'api_type') 
    my_pred_end = GetInfo(df, row, 'pred_end')

    # if it is transfer api 
    if my_type in ['h2d', 'd2h']:
        # update the bytes
        df = UpdateCell(df, row, 'bytes_done', GetInfo(df, row, 'size_kb')) 
        df = UpdateCell(df, row, 'bytes_left', 0) 

    # use pred_end to update the end time
    #df = UpdateCell(df, row, 'end', my_pred_end)

    # move current pos to the end time
    df = UpdateCell(df, row, 'current_pos', my_pred_end) 
    df = UpdateCell(df, row, 'status', 'done') 

    return df


