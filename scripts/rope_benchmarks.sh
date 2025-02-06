#/bin/bash

python ../thunder_model_blocks/hf_phi3/phi3_rope.py --seq_lens 2048 --csv
python ../thunder_model_blocks/hf_phi3/phi3_rope.py --seq_lens 4096 --csv
python ../thunder_model_blocks/hf_phi3/phi3_rope.py --seq_lens 8192 --csv
python ../thunder_model_blocks/hf_phi3/phi3_rope.py --seq_lens 12288 --csv
python ../thunder_model_blocks/hf_phi3/phi3_rope.py --seq_lens 16384 --csv
python ../thunder_model_blocks/hf_phi3/phi3_rope.py --seq_lens 20480 --csv

python ../thunder_model_blocks/hf_qwen2/qwen2_rope.py --seq_lens 2048 --csv
python ../thunder_model_blocks/hf_qwen2/qwen2_rope.py --seq_lens 4096 --csv
python ../thunder_model_blocks/hf_qwen2/qwen2_rope.py --seq_lens 8192 --csv
python ../thunder_model_blocks/hf_qwen2/qwen2_rope.py --seq_lens 12288 --csv
python ../thunder_model_blocks/hf_qwen2/qwen2_rope.py --seq_lens 16384 --csv
python ../thunder_model_blocks/hf_qwen2/qwen2_rope.py --seq_lens 20480 --csv

python ../thunder_model_blocks/hf_mistral/mistral_rope.py --seq_lens 2048 --csv
python ../thunder_model_blocks/hf_mistral/mistral_rope.py --seq_lens 4096 --csv
python ../thunder_model_blocks/hf_mistral/mistral_rope.py --seq_lens 8192 --csv
python ../thunder_model_blocks/hf_mistral/mistral_rope.py --seq_lens 12288 --csv
python ../thunder_model_blocks/hf_mistral/mistral_rope.py --seq_lens 16384 --csv
python ../thunder_model_blocks/hf_mistral/mistral_rope.py --seq_lens 20480 --csv
