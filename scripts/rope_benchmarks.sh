#/bin/bash

python ../hf_phi3/phi3_rope.py --seq_lens 2048 --csv
python ../hf_phi3/phi3_rope.py --seq_lens 4096 --csv
python ../hf_phi3/phi3_rope.py --seq_lens 8192 --csv
python ../hf_phi3/phi3_rope.py --seq_lens 12288 --csv
python ../hf_phi3/phi3_rope.py --seq_lens 16384 --csv
python ../hf_phi3/phi3_rope.py --seq_lens 20480 --csv

python ../hf_qwen2/qwen2_rope.py --seq_lens 2048 --csv
python ../hf_qwen2/qwen2_rope.py --seq_lens 4096 --csv
python ../hf_qwen2/qwen2_rope.py --seq_lens 8192 --csv
python ../hf_qwen2/qwen2_rope.py --seq_lens 12288 --csv
python ../hf_qwen2/qwen2_rope.py --seq_lens 16384 --csv
python ../hf_qwen2/qwen2_rope.py --seq_lens 20480 --csv

python ../hf_mistral/mistral_rope.py --seq_lens 2048 --csv
python ../hf_mistral/mistral_rope.py --seq_lens 4096 --csv
python ../hf_mistral/mistral_rope.py --seq_lens 8192 --csv
python ../hf_mistral/mistral_rope.py --seq_lens 12288 --csv
python ../hf_mistral/mistral_rope.py --seq_lens 16384 --csv
python ../hf_mistral/mistral_rope.py --seq_lens 20480 --csv
