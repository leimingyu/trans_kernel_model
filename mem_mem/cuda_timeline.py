import warnings
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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

    if ssm_coef == None:
        ssm_coef = 0.0
    if dsm_coef == None:
        dsm_coef = 0.0

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


def plot_trace(df_trace, savefig=False):
    """
    Plot the cuda timeline from trace file (pandas dataframe).
    """
    streamList = []

    # read the number of unique streams
    stream_id_list = df_trace['Stream'].unique()
    stream_id_list = stream_id_list[~np.isnan(stream_id_list)] # remove nan

    num_streams = len(stream_id_list)

    for i in xrange(num_streams):
        streamList.append(streams())

    start_coef, duration_coef = time_coef_ms(df_trace)

    # read row by row
    for rowID in xrange(1, df_trace.shape[0]):
        #  extract info from the current row
        stream_id, api_type, start_time_ms, end_time_ms, _ = read_row(df_trace.iloc[[rowID]], start_coef, duration_coef)

        # find out index of the stream
        sid, = np.where(stream_id_list==stream_id)

        # add the start/end time for different api calls
        if api_type == 'h2d':
            streamList[sid].h2d.append(transfer(start_time_ms, end_time_ms))
        elif api_type == 'd2h':
            streamList[sid].d2h.append(transfer(start_time_ms, end_time_ms))
        elif api_type == 'kernel':
            streamList[sid].kernel.append(transfer(start_time_ms, end_time_ms))
        else:
            print("Unknown. Error!")


    fig, ax = plt.subplots()

    # each bar will be 1 in height, the interval between centers of each bar is 2
    # for example, bar 1 is at 1 with width 1 (1, 1), then bar 2 is at 3 with width 1 (3, 1), so on and so forth

    transfer_color = '#C5EDEE'
    kernel_color = '#D2E307'

    stream_num = len(streamList)

    ylim_max = 1 + stream_num * 2.0

    stream_tag_pos =  []
    stream_tag = []

    for i in xrange(stream_num):
        ii = i + 1

        bar_center = ylim_max - ii * 2.0
        bar_loc = (bar_center, 1)  # width 1

        # y lable
        stream_tag_pos.append(bar_center + 0.5) # 0.5 interv
        stream_tag.append('stream-'+ str(i))

        current_stream = streamList[i]

        api_call_seq = []
        api_color_seq = []

        # h2d
        for j in xrange(len(current_stream.h2d)):
            start_time = current_stream.h2d[j].start_time_ms
            duration = current_stream.h2d[j].end_time_ms - current_stream.h2d[j].start_time_ms # add start and duration
            api_call_seq.append((start_time, duration))

            api_color_seq.append(transfer_color) # add the color for bar

            # pos for the annotation: shift left 0.0015 in the middle of the bar
            ax.annotate('h2d', (start_time + duration * 0.35, bar_center + 0.25), fontsize=10)


        # d2h
        for j in xrange(len(current_stream.d2h)):
            start_time = current_stream.d2h[j].start_time_ms
            duration = current_stream.d2h[j].end_time_ms - current_stream.d2h[j].start_time_ms
            api_call_seq.append((start_time, duration))

            api_color_seq.append(transfer_color)

            # pos for the annotation:
            ax.annotate('d2h', (start_time + duration * 0.35, bar_center + 0.25), fontsize=10)

        # kernel
        for j in xrange(len(current_stream.kernel)):
            start_time = current_stream.kernel[j].start_time_ms
            duration = current_stream.kernel[j].end_time_ms - current_stream.kernel[j].start_time_ms
            api_call_seq.append((start_time, duration))

            api_color_seq.append(kernel_color)

            # kernel annotation
            ax.annotate('K', (start_time + duration * 0.35, bar_center + 0.25), fontsize=10)

        # add the bar to the plot for current stream
        ax.broken_barh(api_call_seq,
                bar_loc,
                facecolors=api_color_seq)

    # plot
    ax.set_ylim(0, ylim_max)
    ax.set_xlabel('timeline (ms)')
    ax.set_yticks(stream_tag_pos)
    ax.set_yticklabels(stream_tag)

    aspectratio=0.2
    ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
    ax.set_aspect(ratio_default*aspectratio)

    plt.show()

    if savefig == True:
        fig.savefig('timeline_output.pdf')

#--------------------------------------------------------
# Plot the timeline using a list of dataframe trace.
#--------------------------------------------------------
def plot_cke_list(df_cke_list, savefig = True):
    """
    Plot cuda timeline from the cke list.
    Each cuda stream trace is one list of the list array.
    Each trace has columns : api_type, start, end, duration.
    """
    stream_num = len(df_cke_list)
    # print num_streams
        
    fig, ax = plt.subplots()
    
    h2d_color = '#C5EDEE'
    d2h_color = '#84d3ed'
    kern_color = '#D2E307'
    
    ylim_max = 1 + stream_num * 2.0
    
    stream_tag_pos =  []
    stream_tag = []
    
    for i in xrange(stream_num):
        ii = i + 1
    
        bar_center = ylim_max - ii * 2.0
        bar_loc = (bar_center, 1)  # width 1
    
        # y lable 
        stream_tag_pos.append(bar_center + 0.5) # 0.5 interv
        stream_tag.append('stream-'+ str(i))

        current_stream = df_cke_list[i]

        api_call_seq = []
        api_color_seq = []
        
        for index, row in current_stream.iterrows():
            if row.api_type == 'h2d':
                start_time = row.start
                duration = row.end - start_time
                api_call_seq.append((start_time, duration))
                api_color_seq.append(h2d_color)
                ax.annotate('h2d', (start_time + duration * 0.35, bar_center + 0.25), fontsize=10)
                
            if row.api_type == 'd2h':
                start_time = row.start
                duration = row.end - start_time
                api_call_seq.append((start_time, duration))
                api_color_seq.append(d2h_color)
                ax.annotate('d2h', (start_time + duration * 0.35, bar_center + 0.25), fontsize=10)
                
            if row.api_type == 'kern':
                start_time = row.start
                duration = row.end - start_time
                api_call_seq.append((start_time, duration))
                api_color_seq.append(kern_color)
                ax.annotate('k', (start_time + duration * 0.35, bar_center + 0.25), fontsize=10)
    
        # add the bar to the plot for current stream
        ax.broken_barh(api_call_seq,
               bar_loc, 
               facecolors=api_color_seq)
    #-----------
    # plot all the stream trace
    #-----------
    ax.set_ylim(0, ylim_max)
    ax.set_xlabel('timeline (ms)')
    ax.set_yticks(stream_tag_pos)
    ax.set_yticklabels(stream_tag)

    aspectratio=0.2
    ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
    ax.set_aspect(ratio_default*aspectratio)

    plt.show()

    if savefig == True:
        fig.savefig('timeline_ckelist.pdf')
