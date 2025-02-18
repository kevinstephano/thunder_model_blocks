# Thunder Model Blocks for Performance Debugging

## Options
* `--thunder_trace`: Dumps Forward and Backward Thunder traces.
* `--nvfuser_repro`: Dumps nvFuser python script repros.
* `--nsys`: Turns off torch.profiler usage to allow for NSight Systems profiling.
* `--execs`: Allows you to specify a subset of executors like Thunder-nvFuser.

## To run
```
python thunder_model_blocks/[Model Dir]/[Model Name].py
```

## To install
```
# Please use `-e` especially if you are modifying the script
pip install . [-e]
```
