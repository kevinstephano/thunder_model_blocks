#!/bin/bash

seqlens="1024 2048 4096 8192 12288 16384 20480"
executors="Torch-Eager Thunder-torch.compile Thunder-nvFuser"
models=("phi3" "qwen2" "mistral")

for benchmark in $models; do
  cmd="python ../${benchmark}/${benchmark}_embedding.py --seq_lens ${seqlens} --execs ${executors}"
  eval $cmd
done
