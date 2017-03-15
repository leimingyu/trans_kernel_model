# trans_kernel_model
integrate data transfer plus cke model

# Author
Leiming Yu

* Email: ylm@ece.neu.edu
* Twitter: @yu_leiming
* Blog: http://www1.coe.neu.edu/~ylm/

# Description
https://github.com/3upperm2n/h2d_overlapping
Given the single stream profiling trace, predicting the effects of running the same kernel with same data size using multiple cuda streams.

## step 1: single stream trace
we have 3 different cuda api calls: 
* h2d 
* kernel 
* d2h

If there is multiple h2d calls, there is launching overhead between them. (that could be coming from checking whether the previous api call ends or not). we can name it as **h2d_ovhd**

For kernel api, there are **host2kern_ovhd and kern_ovhd**  which mean that the overhead of launching kernel after h2d and the overhead of launching kernel after kernel.

For d2h api, there are **kern2host_ovhd and d2h_ovhd** which stand for the overhead of launching d2h api after kernel call, and the overhead of launching d2h after d2h call.

To sum up, we have the api call sequence for the single stream case.
```python
H2D + (h2d_num - 1) * (h2d_ovhd + H2D) + 
host2kern_ovhd + Kernel + (knum - 1) * (kern_ovhd + Kernel) +
kern2host_ovhd + D2H + (d2h_num - 1) * (d2h_ovhd + D2H)
```

## step 2: model data transfer H2D
We know that h2d api calls from different streams are launched in order. [check this](https://github.com/3upperm2n/h2d_overlapping)

Until the h2d transfer time reaches a limit (which can be benchmarked), the second stream will not wait for the previous stream finishes all the h2d calls. We call this limit as **trans_ovlp_th**.

Based on the stream-0 (a single stream) profiling trace with all the api call timing, we add on the stream-1 with the same timing as the stream-0.

If the two kernels in both streams have concurrency, we adjust the kernel execution time by adopting the avg block runtime.

If the d2h (stream-0) and h2d (stream-1) have concurrency, we assume there is no performance penalty since pcie-3 is full-duplex.

If the 3rd stream are launched, we schedule the h2d as the previous scheme, similar to stream-1. However, when there is h2d concurrency with prevous streams, even though it is launched, it will wait till one of the api call finishes before it **actually** transfers the data.

Here is an example.
<image src="Figs/mem_mem_trace.png" height="250px">


## step 3: model concurrent kernel
When the (N-1) stream comes in, if it had concurrency with previous stream (N-2), it will also check the timing with the streams before to adjust the runtime for N kernels.

At the beginning, we can offset the starting timing to 0, just for easy-reading.

We need first check whether the data transfer time exceeds the h2d_ovlp_threshold. If it exceeds, the second stream api will start immediately after the threshold. Otherwise, it will wait all the h2d api calls to finish, then to start the the data transfer for the comming stream.

Then, we can ajust the starting point for each cuda stream. Add all the stream trace to the full trace table.
