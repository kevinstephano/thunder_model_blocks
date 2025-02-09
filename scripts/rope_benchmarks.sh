#/bin/bash

python ../hf_phi3/phi3_rope.py --seq_lens 1024 2048 4096 8192 12288 16384 20480 --execs Thunder-torch.compile Thunder-nvFuser

python ../hf_qwen2/qwen2_rope.py --seq_lens 1024 2048 4096 8192 12288 16384 20480 --execs Thunder-torch.compile Thunder-nvFuser

python ../hf_mistral/mistral_rope.py --seq_lens 1024 2048 4096 8192 12288 16384 20480 --execs Thunder-torch.compile Thunder-nvFuser
