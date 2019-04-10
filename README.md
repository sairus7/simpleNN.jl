# simpleNN
## Manual implementation of simple neural network with MNIST dataset

This is an updated version of neural network code from the original question by Anders, answered by Chris Rackauckas: 
[macOS Python with numpy faster than Julia in training neural network](https://stackoverflow.com/questions/49719076/macos-python-with-numpy-faster-than-julia-in-training-neural-network/49724611)

There are four versions of this code:
* v1 - original version includes all recommendations from the answer, with operations changed to support Julia 1.1, and some minor changes to variable naming, styling, etc.
* v2 - load data and batches format changed to matrix instead of array of tuples (loaddata, vectorize, etc.), first activation layer removed, no need to copy input data to it.
* v3 - batch run converted into matrix operations, added batch_size preallocation for nn layers.
* v4 - network structure rework: split into 3 different structures: `network_v4`, `batch_trainer` and `batch_tester` for preallocation, the whole run time is faster due to preallocation for evaluation.

There is a performance increase by an order (!) of magnitude due to transition to matrix operations in batches. 
On my machine:
* v1 and v2 take about 10 seconds per epoch
* v2 and v3 take about 1.1 seconds per epoch