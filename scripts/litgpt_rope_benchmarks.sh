#!/bin/bash

seqlens="1024 2048 4096 8192 12288 16384 20480 24576 28672 32768"
executors="Torch-Eager torch.compile Thunder-torch.compile Thunder-nvFuser"
models=("google/gemma-2-9b-it" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3-8B-Instruct" "microsoft/Phi-3.5-mini-instruct")
d=$(pwd)

for benchmark in ${models[@]}; do
  cmd="python $d/thunder_model_blocks/litgpt/litgpt_rope.py --seq_lens ${seqlens} --model ${benchmark} --execs ${executors} --csv"
  eval $cmd
done
