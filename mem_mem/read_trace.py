import pandas as pd
import numpy as np
import operator


class transfer():
    def __init__(self, start=0.0,end=0.0):
        self.start_time_ms = start
        self.end_time_ms = end


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


class streams():
    def __init__(self):
        self.h2d = []
        self.d2h = []
        self.kernel = []
        self.kernel_info = []


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


# read data for the current row
def read_row(df_row, start_coef_ms, duration_coef_ms, ssm_coef = None, dsm_coef = None):
    start_time_ms = float(df_row['Start']) * start_coef_ms

    end_time_ms = start_time_ms + float(df_row['Duration']) * duration_coef_ms

    stream_id = int(df_row['Stream'])

    api_name = df_row['Name'].to_string()

    kernelinfo = KernConfig()

    if "DtoH" in api_name:
        api_type = 'd2h'
    elif "HtoD" in api_name:
        api_type = 'h2d'
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

    return stream_id, api_type, start_time_ms, end_time_ms, kernelinfo


def trace2dataframe(trace_file):
    """
    read the trace file into dataframe using pandas
    """
    # There are max 17 columns in the output csv
    col_name = ["Start","Duration","Grid X","Grid Y","Grid Z","Block X","Block Y","Block Z","Registers Per Thread","Static SMem","Dynamic SMem","Size","Throughput","Device","Context","Stream","Name"]

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


def get_stream_info(df_trace):
    """
    read dataframe into stream list which contains the h2d/d2h/kernel star and end time in ms.
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

    # read row by row
    for rowID in xrange(1, df_trace.shape[0]):
        #  extract info from the current row
        stream_id, api_type, start_time_ms, end_time_ms, kerninfo = read_row(df_trace.iloc[[rowID]], start_coef, duration_coef, ssm_coef, dsm_coef)

        # find out index of the stream
        sid, = np.where(stream_id_list==stream_id)

        # add the start/end time for different api calls
        if api_type == 'h2d':
            streamList[sid].h2d.append(transfer(start_time_ms, end_time_ms))
        elif api_type == 'd2h':
            streamList[sid].d2h.append(transfer(start_time_ms, end_time_ms))
        elif api_type == 'kernel':
            streamList[sid].kernel.append(transfer(start_time_ms, end_time_ms))
            streamList[sid].kernel_info.append(kerninfo) # add the kernel info
        else:
            print "Unknown. Error."

    return streamList


def check_kernel_ovlprate(trace_file):
    """
    Read the trace file and figure out the overlapping rate for the two kernel execution.
    """
    # read data from the trace file
    df_trace = trace2dataframe(trace_file)

    # extract stream info
    streamList = get_stream_info(df_trace)

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
        stream_id, api_type, start_time_ms, end_time_ms, _ =  read_row(df_trace.iloc[[rowID]], start_coef, duration_coef)

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
