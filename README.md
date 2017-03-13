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

## step 2: 
