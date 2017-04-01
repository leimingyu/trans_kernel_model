import pandas as pd
import numpy as np
from math import *
import operator
import sys

from model_param import *


#class DeviceInfo():
#    def __init__(self, sm_num=0, sharedmem_per_sm=0, reg_per_sm=0, 
#            maxthreads_per_sm=0):
#        self.sm_num = sm_num
#        self.sharedmem_per_sm = sharedmem_per_sm # bytes
#        self.reg_per_sm = reg_per_sm
#        self.maxthreads_per_sm = maxthreads_per_sm


#class KernConfig():
#    def __init__(self,
#                 grid_x = 0, grid_y = 0, grid_z = 0,
#                 blk_x = 0, blk_y = 0, blk_z = 0,
#                 regs_per_thread = 0, sm_per_block = 0):
#        self.grid_x = grid_x
#        self.grid_y = grid_y
#        self.grid_z = grid_z
#        self.blk_x = blk_x
#        self.blk_y = blk_y
#        self.blk_z = blk_z
#        self.regs_per_thread = regs_per_thread
#        self.sm_per_block = sm_per_block


#class KernelInfo():
#    def __init__(self, blockDim=0, gridDim=0, reg_per_thread=0, 
#            sharedmem_per_blk=0, runtime_ms = 0, avg_blk_time = 0):
#        self.blockDim = blockDim
#        self.gridDim = gridDim
#        self.reg_per_thread = reg_per_thread
#        self.sharedmem_per_blk =  sharedmem_per_blk
#        self.runtime_ms = runtime_ms
#        self.avg_blk_time = avg_blk_time

class transfer():
    def __init__(self, start=0.0, end=0.0, trans_size = 0.0):
        self.start_time_ms = start
        self.end_time_ms = end
        self.size = trans_size 


class streams():
    def __init__(self):
        self.h2d = []
        self.d2h = []
        self.kernel = []
        self.kernel_info = []



#------------------------------------------------------------------------------
# Use ms for timing
#------------------------------------------------------------------------------
def time_coef_ms(df_trace):
    rows, cols = df_trace.shape

    start_unit = df_trace['Start'].iloc[0]
    duration_unit = df_trace['Duration'].iloc[0]

    start_coef =  1.0
    if start_unit == 's':
        start_coef = 1e3
    if start_unit == 'us':
        start_coef = 1e-3

    duration_coef =  1.0
    if duration_unit == 's':
        duration_coef = 1e3
    if duration_unit == 'us':
        duration_coef = 1e-3

    return start_coef, duration_coef


#------------------------------------------------------------------------------
# Use bytes for shared memory 
#------------------------------------------------------------------------------
def sm_coef_bytes(df_trace):
    ssm_unit = df_trace['Static SMem'].iloc[0]
    dsm_unit = df_trace['Dynamic SMem'].iloc[0]

    ssm_coef = 1.0
    if ssm_unit == 'KB':
        ssm_coef = 1e3
    if ssm_unit == 'MB':
        ssm_coef = 1e6

    dsm_coef = 1.0
    if dsm_unit == 'KB':
        dsm_coef = 1e3
    if dsm_unit == 'MB':
        dsm_coef = 1e6

    return ssm_coef, dsm_coef


#------------------------------------------------------------------------------
# Use KB for data transfer 
#------------------------------------------------------------------------------
def trans_coef_kb(df_trace):
    size_unit = df_trace['Size'].iloc[0]

    coef = 0.0 # KB

    if size_unit == 'B':
        coef = 1e-3
    elif size_unit == 'MB':
        coef = 1e3
    elif size_unit == 'GB':
        coef = 1e6
    elif size_unit == 'KB':
        coef = 1.0 # KB 
    else:
        sys.stderr.write('Unknown Size Unit.\n')

    return coef 


#------------------------------------------------------------------------------
# Read the current row of the input dataframe trace.
# Return : stream_id, api_type, start, end, kerninfo   
#------------------------------------------------------------------------------
def read_row(df_row, start_coef_ms, duration_coef_ms, ssm_coef = None, 
        dsm_coef = None, trans_coef = 1.0):
    # output paramters
    stream_id = int(df_row['Stream'])
    start_time_ms = float(df_row['Start']) * start_coef_ms
    end_time_ms = start_time_ms + float(df_row['Duration']) * duration_coef_ms
    api_name = df_row['Name'].to_string()
    kernelinfo = KernConfig()
    trans_kb = None

    if ssm_coef == None:
        ssm_coef = 0.0
    if dsm_coef == None:
        dsm_coef = 0.0

    if "DtoH" in api_name:
        api_type = 'd2h'
        trans_kb = float(df_row.Size) * trans_coef # d2h size in KB
    elif "HtoD" in api_name:
        api_type = 'h2d'
        trans_kb = float(df_row.Size) * trans_coef # h2d size in KB
    else:
        api_type = 'kernel'
        # read kernel and update the info
        grid_x = float(df_row['Grid X'])
        grid_y = float(df_row['Grid Y'])
        grid_z = float(df_row['Grid Z'])
        blk_x = float(df_row['Block X'])
        blk_y = float(df_row['Block Y'])
        blk_z = float(df_row['Block Z'])
        regs_per_thread = float(df_row['Registers Per Thread'])

        static_sm = float(df_row['Static SMem'])
        dynamic_sm = float(df_row['Dynamic SMem'])
        sm_per_block = static_sm * ssm_coef + dynamic_sm * dsm_coef

        kernelinfo.blk_x = blk_x
        kernelinfo.blk_y = blk_y
        kernelinfo.blk_z = blk_z
        kernelinfo.grid_x = grid_x
        kernelinfo.grid_y = grid_y
        kernelinfo.grid_z = grid_z

        kernelinfo.regs_per_thread = regs_per_thread
        kernelinfo.sm_per_block = sm_per_block

    return stream_id, api_type, start_time_ms, end_time_ms, kernelinfo, trans_kb


#------------------------------------------------------------------------------
#  Read current row of the dataframe, return timing and transfer_size
#------------------------------------------------------------------------------
def read_row_for_timing(df_row, start_coef_ms, duration_coef_ms, trans_coef=0.0):
    """
    Read the current row for the dataframe, extracting timing only.
    """
    # parameters
    stream_id = int(df_row['Stream'])
    api_name = df_row['Name'].to_string()
    start_time_ms = float(df_row['Start']) * start_coef_ms
    end_time_ms = start_time_ms + float(df_row['Duration']) * duration_coef_ms

    trans_kb = 0.0 

    if "DtoH" in api_name:
        api_type = 'd2h'
        trans_kb = float(df_row.Size) * trans_coef # d2h size in KB
    elif "HtoD" in api_name:
        api_type = 'h2d'
        trans_kb = float(df_row.Size) * trans_coef # d2h size in KB
    else:
        api_type = 'kern'

    return stream_id, api_type, start_time_ms, end_time_ms, trans_kb


#------------------------------------------------------------------------------
#  Read csv trace to dataframe in pandas. 
#------------------------------------------------------------------------------
def Trace2dataframe(trace_file):
    """
    read the trace file into dataframe using pandas
    """
    # There are max 17 columns in the output csv
    col_name = ["Start","Duration","Grid X","Grid Y","Grid Z","Block X",
            "Block Y","Block Z","Registers Per Thread","Static SMem",
            "Dynamic SMem","Size","Throughput","Device","Context","Stream","Name"]
    df_trace = pd.read_csv(trace_file, names=col_name, engine='python')
    rows_to_skip = 0
    # find out the number of rows to skip
    for index, row in df_trace.iterrows():
        if row['Start'] == 'Start':
            rows_to_skip = index
            break
    # read the input csv again
    df_trace = pd.read_csv(trace_file, skiprows=rows_to_skip)

    return df_trace


#------------------------------------------------------------------------------
#  Read csv trace to dataframe in pandas. 
#------------------------------------------------------------------------------
def Get_stream_info(df_trace):
    """
    Read dataframe into stream list,
    which contains the h2d/d2h/kernel start and end time in ms.
    """
    streamList = []

    # read the number of unique streams
    stream_id_list = df_trace['Stream'].unique()
    stream_id_list = stream_id_list[~np.isnan(stream_id_list)] # remove nan

    num_streams = len(stream_id_list)

    for i in xrange(num_streams):
        streamList.append(streams())

    start_coef, duration_coef = time_coef_ms(df_trace)

    ssm_coef, dsm_coef = sm_coef_bytes(df_trace)

    trans_coef = trans_coef_kb(df_trace) # normalize the transfer size to KB

    # read row by row
    for rowID in xrange(1, df_trace.shape[0]):
        #  extract info from the current row
        stream_id, api_type, start_time_ms, end_time_ms, kerninfo, Tkb = \
                read_row(df_trace.iloc[[rowID]], start_coef, duration_coef, 
                        ssm_coef, dsm_coef, trans_coef)

        # find out index of the stream
        sid, = np.where(stream_id_list==stream_id)

        # add the start/end time for different api calls
        if api_type == 'h2d':
            streamList[sid].h2d.append(transfer(start_time_ms, end_time_ms, Tkb))
        elif api_type == 'd2h':
            streamList[sid].d2h.append(transfer(start_time_ms, end_time_ms, Tkb))
        elif api_type == 'kernel':
            # kernel runtime
            streamList[sid].kernel.append(transfer(start_time_ms, end_time_ms))
            # kernel info 
            streamList[sid].kernel_info.append(kerninfo)
        else:
            print "Unknown. Error."

    return streamList


def check_kernel_ovlprate(trace_file):
    """
    Read the trace file and figure out the overlapping rate for the two kernel execution.
    """
    # read data from the trace file
    df_trace = Trace2dataframe(trace_file)

    # extract stream info
    streamList = Get_stream_info(df_trace)

    # check kernel overlapping
    preK_start = streamList[0].kernel[0].start_time_ms
    preK_end = streamList[0].kernel[0].end_time_ms

    curK_start = streamList[1].kernel[0].start_time_ms
    curK_end = streamList[1].kernel[0].end_time_ms

    preK_runtime = preK_end - preK_start
    curK_runtime = curK_end - curK_start

    ovlp_duration = preK_end - curK_start
    ovlp_ratio = ovlp_duration / preK_runtime

#    if curK_start >= preK_start and curK_start <= preK_end:
#        print('concurrent kernel execution :\n\t stream-prev {} ms \n\t stream-cur {} ms'
#        '\n\t overlapping {} ms \n\t ovlp ratio (based on prev stream) {}%'\
#              .format(preK_runtime, curK_runtime, ovlp_duration, ovlp_ratio))

    cke_time_ms = curK_end - preK_start

    return ovlp_ratio, cke_time_ms


def get_kernel_time_from_trace(df_trace):
    """
    Read kernel time from trace.
    """
    # read the number of unique streams
    stream_id_list = df_trace['Stream'].unique()
    stream_id_list = stream_id_list[~np.isnan(stream_id_list)] # remove nan

    start_coef, duration_coef = time_coef_ms(df_trace)

    ssm_coef, dsm_coef = sm_coef_bytes(df_trace)

    kernel_time_dd = {}

    # read row by row
    for rowID in xrange(1, df_trace.shape[0]):
        #  extract info from the current row
        stream_id, api_type, start_time_ms, end_time_ms, _ =  \
                read_row(df_trace.iloc[[rowID]], start_coef, duration_coef)

        # find out index of the stream
        sid, = np.where(stream_id_list == stream_id)

        sid = int(sid)
        # find out the duration for kernel
        if api_type == 'kernel':
            duration = end_time_ms - start_time_ms
            kernel_time_dd[sid] = duration

    return kernel_time_dd


def kernel_slowdown(s1_kernel_dd, s2_kernel_dd):
    slow_down_ratio_list = []
    for key, value in s2_kernel_dd.items():
        v_s1 = s1_kernel_dd[0]
        slow_down_ratio_list.append(value / float(v_s1))
    return slow_down_ratio_list



# -----------------------------------------------------------------------------
# get timing trace from the dataframe
# -----------------------------------------------------------------------------
def Get_timing_from_trace(df_trace):
    """
    Extract api call and timings from trace of single stream. Return dataframe.
    """
    stream_id_list = df_trace['Stream'].unique()
    stream_id_list = stream_id_list[~np.isnan(stream_id_list)] # remove nan
    num_streams = len(stream_id_list)

    streamList = [[] for i in range(num_streams)]

    start_coef, duration_coef = time_coef_ms(df_trace) # convert time to ms
    trans_coef = trans_coef_kb(df_trace) # normalize the transfer size to KB

    for rowID in xrange(1, df_trace.shape[0]):
        stream_id, api_type, start_time_ms, end_time_ms, trans_kb = \
                read_row_for_timing(df_trace.iloc[[rowID]], start_coef, 
                        duration_coef, trans_coef)

        # find out index of the stream
        sid, = np.where(stream_id_list==stream_id)
        # print("{} {} : {} - {}".format(sid, api_type, start_time_ms, end_time_ms))
        streamList[sid].append([api_type, start_time_ms, end_time_ms, trans_kb])

    # api timing for current stream
    df_stream = pd.DataFrame(columns=['stream', 'api_type', 'start', 'end', 'size'])

    # for each stream: update the trace
    for sid in range(num_streams):
        current_stream_list = streamList[sid]
        rows = len(current_stream_list)

        for i in range(rows):
            curr_api = current_stream_list[i][0]
            # print curr_api
            curr_start = current_stream_list[i][1]
            curr_end   = current_stream_list[i][2]
            curr_size  = current_stream_list[i][3]

            # add current api
            df_stream = df_stream.append({'stream': sid,
                'api_type': curr_api, 
                'start': curr_start, 
                'end': curr_end,
                'size': curr_size}, ignore_index=True)

    df_stream['duration'] = df_stream['end'] - df_stream['start']
    return df_stream


# ------------------------
# Get total runtime from the trace.
# trace table columns: stream 	api_type 	start 	end 	duration
# ------------------------
def GetTotalRuntime(df_trace_new):
    # case 1) the input is a list of dataframe
    if isinstance(df_trace_new, list):
        #print('it is list')
        num_element = len(df_trace_new)
        min_t = float(df_trace_new[0].start.min()) 
        max_t = float(df_trace_new[0].end.max())
        for i in range(1, num_element):
            cur_min = float(df_trace_new[i].start.min())
            cur_max = float(df_trace_new[i].end.max())
            if cur_min < min_t: min_t = cur_min
            if cur_max > max_t: max_t = cur_max

        return cur_max - cur_min

    # case 2) the input is a single dataframe
    if isinstance(df_trace_new, pd.DataFrame):
        #print('it is dataframe')
        return float(df_trace_new.end.max()) - float(df_trace_new.start.min())


# -----------------------------------------------------------------------------
# Reset the start time to zero for the input dataframe trace.
# -----------------------------------------------------------------------------
def Reset_starting(df_org):
    df_trace = df_org.copy(deep=True)
    offset = df_trace.start[0]
    #print offset
    df_trace.start = df_trace.start - offset
    df_trace.end = df_trace.end - offset
    return df_trace


def find_h2ds_timing(df_trace):
    """
    find the h2d start and end for the current stream
    """
    h2ds_begin = 0
    for index, row in df_trace.iterrows():
        if row['api_type'] == 'h2d':
            h2ds_begin = row.start # read the 1st h2d start
            break

    h2ds_end = 0
    for index, row in df_trace.iterrows():
        if row['api_type'] == 'h2d':
            h2ds_end = row.end # read the h2d end, till the kernel is met

        if row['api_type'] == 'kern':
            break;

    return h2ds_begin, h2ds_end


def find_kern_timing(df_trace):
    """
    find the h2d start and end for the current stream
    """
    kern_begin = 0
    kern_end = 0
    for index, row in df_trace.iterrows():
        if row['api_type'] == 'kern':
            kern_begin = row.start
            kern_end = row.end
            break;

    return kern_begin, kern_end

def find_d2h_timing(df_trace):
    """
    find the h2d start and end for the current stream
    """
    Tbegin = 0
    Tend = 0
    for index, row in df_trace.iterrows():
        if row['api_type'] == 'd2h':
            Tbegin = row.start
            Tend = row.end
            break;
    return Tbegin, Tend




#------------------------------------------------------------------------------
# Read the kernel information from trace file. 
# Warning: assume there is only one kernel in the cuda stream
#       we need to generate data dict to store the kernel_info by the signature
#------------------------------------------------------------------------------
def GetKernelInfo(df_trace, Gpu):
    # extract kernel info
    streaminfo = Get_stream_info(df_trace)
    # print('streams : {}'.format(len(streaminfo)))
    # print len(streaminfo[0].kernel)
    # print len(streaminfo[0].kernel_info)

    current_kern_info = streaminfo[0].kernel_info[0]
    grid_dim = float(current_kern_info.grid_x) * float(current_kern_info.grid_y) * float(current_kern_info.grid_z)
    block_dim = float(current_kern_info.blk_x) * float(current_kern_info.blk_y) * float(current_kern_info.blk_z)
    reg_per_thread = float(current_kern_info.regs_per_thread)
    sm_per_blk = float(current_kern_info.sm_per_block)

    # kernel runtime in ms
    current_kern =  streaminfo[0].kernel[0]
    kern_runtime_ms = float(current_kern.end_time_ms) - float(current_kern.start_time_ms)

    kernel = KernelInfo()
    kernel.blockDim = block_dim
    kernel.gridDim = grid_dim
    kernel.reg_per_thread = reg_per_thread
    kernel.sharedmem_per_blk = sm_per_blk
    kernel.runtime_ms = kern_runtime_ms

    kernel.avg_blk_time = AvgBlkTime(Gpu, kernel)

    return kernel

def DumpKernelInfo(kernel):
    print('Kernel Info'
        '\n\tblockDim {}'
        '\n\tgridkDim {}'
        '\n\tregs {}'
        '\n\tshared memory{}'
        '\n\truntime (ms) {}'
        '\n\taverage block execution time (ms) {}'.format(kernel.blockDim,
        kernel.gridDim,
        kernel.reg_per_thread,
        kernel.sharedmem_per_blk,
        kernel.runtime_ms,
        kernel.avg_blk_time))
