#!/bin/bash

seqlens="4096"
#executors="Torch-Eager torch.compile Thunder-torch.compile Thunder-nvFuser"
executors="Torch-Eager"
d=$(pwd)

# Specify the path to your text file containing names
file="scripts/huggingface_models.txt"

# Check if the file exists
if [ ! -f "$file" ]; then
  echo "Error: File '$file' not found."
  exit 1
fi

# Iterate over each line in the file
while IFS= read -r benchmark; do
  cmd="python $d/thunder_model_blocks/auto_model/auto_model.py --seq_lens ${seqlens} --model ${benchmark} --execs ${executors} --two_layers --csv"
  eval $cmd
done < "$file"
