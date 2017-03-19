import pandas as pd
import numpy as np
import operator
import sys


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
