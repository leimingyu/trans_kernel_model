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

                

    return df_all_api 
