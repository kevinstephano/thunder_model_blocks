#!/bin/bash

seqlens="4096"
executors="Torch-Eager Thunder-torch.compile Thunder-nvFuser"
models=("phi3" "qwen2" "mistral")
d=$(pwd)

for benchmark in ${models[@]}; do
  cmd="python $d/thunder_model_blocks/${benchmark}/${benchmark}_attn_block.py --seq_lens ${seqlens} --lora --packed_seqs --execs ${executors}"
  eval $cmd
done
