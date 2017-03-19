import pandas as pd
import numpy as np
import operator
import sys


#------------------------------------------------------------------------------
# Update Cell Function.
# Example:
# (old)  
# df_all_api.iloc[0, df_all_api.columns.get_loc('time_left')] = current_predict_time_left
# (now)
# df_all_api = updateCell(df_all_api, 0, 'time_left', current_predict_time_left)
#------------------------------------------------------------------------------
def UpdateCell(df_all_api, row_id, col_name, val):
    df = df_all_api.copy(deep=True)
    df.iloc[row_id, df.columns.get_loc(col_name)] = val
    return df
