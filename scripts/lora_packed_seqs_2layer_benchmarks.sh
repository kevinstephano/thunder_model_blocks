#!/bin/bash

seqlens="1024 2048 4096 8192 12288 16384 20480"
executors="Torch-Eager Thunder-torch.compile Thunder-nvFuser"
models=("phi3" "qwen2" "mistral" "gemma2" "starcoder2" "llama")
d=$(pwd)

for benchmark in ${models[@]}; do
  cmd="python $d/thunder_model_blocks/${benchmark}/${benchmark}_model.py --seq_lens ${seqlens} --lora --packed_seqs --two_layers --execs ${executors}"
  eval $cmd
done
