# weather_bench_data

1. "Data_downloader_large.py" to download raw data
2. "Data_division.py" to preprocess(downsample, divide into input/target, etc) data
3. "Data_40_delete.py" to sparsify data
4. "Data_merge(time).py" to merge data in time domain

## Data Directory Structure
"weather_bench_data" repo, "metnet3" repo and the dataset directory(will be automatically generated using "Data_downloader_large.py") are assumed to be located in the same root. 
```
RootDIR/
    -weather_bench_data/
    -metnet3/
    -datasets/
        -weather_bench/
        -...
        -...
```
            