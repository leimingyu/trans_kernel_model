import pandas as pd
import numpy as np
import operator
import sys
from math import *


#------------------------------------------------------------------------------
# Update Cell Function.
# 
# Example (old)  
# df_all_api.iloc[0, df_all_api.columns.get_loc('time_left')] = current_predict_time_left
# (now)
# df_all_api = updateCell(df_all_api, 0, 'time_left', current_predict_time_left)
#------------------------------------------------------------------------------
def UpdateCell(df_all_api, row_id, col_name, val):
    df = df_all_api.copy(deep=True)
    df.iloc[row_id, df.columns.get_loc(col_name)] = val
    return df


#------------------------------------------------------------------------------
# Update transfer timing: bytes_left, time_left, predicted end time
#------------------------------------------------------------------------------
def UpdateTranTime(df_all, row_list, ways = 1.0):
    """
    From the input dataframe, adjust the row info, depending on the concurrency.
    """
    df_all_api = df_all.copy(deep=True)

    cc = ways # concurrency

    for i in row_list:
        # the bandwidth is shared among all the concurrent transfer
        bw = df_all_api.iloc[i]['bw'] / cc
        # predict the transfer time based on the bandwidth
        cur_pred_time_left = df_all_api.iloc[i]['bytes_left'] / bw
        # update the cell: time_left 
        df_all_api = UpdateCell(df_all_api, i, 'time_left', cur_pred_time_left)
        # update the future ending time
        df_all_api = UpdateCell(df_all_api, i, 'pred_end', 
                cur_pred_time_left + df_all_api.iloc[i]['current_pos'] )
    
    return df_all_api


#------------------------------------------------------------------------------
# Get the time range to check concurrency 
#------------------------------------------------------------------------------
def GetRangeFromWakeStream(df_all_api):
    df_wake = df_all_api.loc[df_all_api.status == 'wake']
    startT = df_wake.current_pos.min()
    endT = df_wake.pred_end.min()
    return [startT, endT]


#------------------------------------------------------------------------------
# Check whether there is other stream for overlapping 
#------------------------------------------------------------------------------
def CheckOtherStream(df_all_api, time_interv):
    df_wake = df_all_api.loc[df_all_api.status == 'wake']
    df_sleep = df_all_api.loc[df_all_api.status <> 'wake']
    
    new_stream_ls = []
    for x in df_sleep.stream_id.unique():
        if x not in df_wake.stream_id.unique():
            new_stream_ls.append(x)
    
    newS = 0
    if new_stream_ls:
       newS = 1 

    return newS

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
def UpdateWakeTiming(df_all, time_interv, cc):
    startT = time_interv[0]
    endT = time_interv[1]
    dur = endT - startT

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
# 
#------------------------------------------------------------------------------
def DoneApiUpdate(df_all_api):
    # copy the input
    df_all = df_all_api.copy(deep=True)
    df_done = df_all.loc[df_all.status == 'done'] # find out which api is done
    done_streams = df_done.stream_id.unique() # np.array

    for x in done_streams:
        df_curr = df_all.loc[df_all.stream_id == x] # the api in order

        prev_start = 0.0
        prev_end = 0.0
        prev_pred_end = 0.0
        prev_status = '' 
        prev_newEnd = 0.0

        count = 0
        for index, row in df_curr.iterrows():
            # record previous timing and status
            if count == 0:
                prev_start = row.start
                prev_end = row.end
                #print('prev_end {}'.format(prev_end))
                prev_pred_end = row.pred_end
                prev_status = row.status

            cur_start = row.start 
            #print('cur_start {}'.format(cur_start))
            cur_end = row.end
            cur_pred_end = row.pred_end
            cur_status = row.status

            #print('count {} : cur_start {}  prev_end {}'.format(count, cur_start, prev_end)) 

            if cur_status == 'done':
                pass # do nothing 
            else:
                # adjust offset according to the previous predicted_end
                ovhd = cur_start - prev_end 
                #print('count {} : ovhd {}'.format(count, ovhd)) 

                if prev_status == 'done':
                    new_start = prev_pred_end + ovhd    # offset with the pred_end
                else:
                    new_start = prev_newEnd + ovhd # with previous new_end

                new_end = new_start + (cur_end - cur_start)  # new start + duration

                # before updating the current record, save the current 
                prev_start = cur_start
                prev_end = cur_end
                prev_pred_end = cur_pred_end
                prev_status = cur_status
                prev_newEnd = new_end # important!

                # update the dataframe record
                #print index
                df_all.set_value(index, 'start', new_start)
                df_all.set_value(index, 'end', new_end)


            # update the count for current iter
            count = count + 1

    return df_all


#------------------------------------------------------------------------------
# Set the target row to be wake status
#------------------------------------------------------------------------------
def SetWake(df_all, r1):
    df_all_api = df_all.copy(deep=True)
    df_all_api = UpdateCell(df_all_api, r1, 'status', 'wake')
    return df_all_api

