Genie and Lamp GPU
===

### Milestone:

* Finish GPU and GPU SCAN with CUDA. And the the top K is selected by ***bucket algorithm***. Details can be found in [https://github.com/imwithye/CUDA-K-Nearest-Neighbors](https://github.com/imwithye/CUDA-K-Nearest-Neighbors). Bucket selection algorithm is published on [http://www.math.grinnell.edu/~blanchaj/Research/Research.html#ggkspaper](http://www.math.grinnell.edu/~blanchaj/Research/Research.html#ggkspaper) and implemented by [Yiwei](http://github.com/imwithye), imwithye@gmail.com

* Finish CPU SCAN algorithm

---

### Running test case:

***Note: Build this project in nsight release mode!***

#### Command line arguments:

1. `--topk` or `-k` sets the value of K. Default value is 128
2. `--dimension` or `-d` sets the value of dimension number. Default value is 256
3. `--query` or `-q` sets the value of query number. Default value is 64

#### Example usage:

```Bash
$ Release/Genie-and-Lamp-GPU -k 32 -d 128 -q 128
$ Release/Genie-and-Lamp-GPU -q 256
```

#### Batch run

Make sure you are in project directory, run

```Bash
[user@host Genie-and-Lamp-GPU] $ ruby runner.rb
```

You may add more test case by editing the arguments list in `runner.rb`:

```Ruby
def cases
  [
  #[topk, dimension, query]
    [2,   8,         2],
    [4,   16,        8],
    [8,   32,        32],
    [16,  64,        128],
    [32,  64,        128],
    [64,  128,       128],
    [128, 128,       256],
    [128, 256,       1028],
  ]
end
```

Here is some example output:

```Bash
==> running Release/Genie-and-Lamp-GPU -k 16 -d 64 -q 128
TOPK = 16 DIMENSIONNUM = 64 QUERYNUM = 128
GPU     : finished with total time : 0.83 with 26 iterations
CPU_SCAN: the time of top-16 in CPU version is:0.35
GPU_SCAN: GPU SCAN Time used: 0.0222176

==> running Release/Genie-and-Lamp-GPU -k 32 -d 64 -q 128
TOPK = 32 DIMENSIONNUM = 64 QUERYNUM = 128
GPU     : finished with total time : 0.81 with 26 iterations
CPU_SCAN: the time of top-32 in CPU version is:0.36
GPU_SCAN: GPU SCAN Time used: 0.0217303

==> running Release/Genie-and-Lamp-GPU -k 64 -d 128 -q 128
TOPK = 64 DIMENSIONNUM = 128 QUERYNUM = 128
GPU     : finished with total time : 1.02 with 31 iterations
CPU_SCAN: the time of top-64 in CPU version is:0.68
GPU_SCAN: GPU SCAN Time used: 0.0297324
```
