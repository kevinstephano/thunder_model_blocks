# Thunder Model Blocks for Performance Debugging

## Options
* `--thunder_trace`: Dumps Forward and Backward Thunder traces.
* `--nvfuser_repro`: Dumps nvFuser python script repros.
* `--nsys`: Turns off torch.profiler usage to allow for NSight Systems profiling.
* `--execs`: Allows you to specify a subset of executors like Thunder-nvFuser.

## To run
### hf_qwen2 Blocks
#### Qwen2 Model
```
python hf_qwen2/qwen2_lora_model.py
```
#### Qwen2 Multihead Attention Block
```
python hf_qwen2/qwen2_lora_attn_block.py
```
#### Qwen2 MLP Block
```
python hf_qwen2/qwen2_lora_mlp_block.py
```
#### Qwen2 Decoder Layer Block
```
python hf_qwen2/qwen2_lora_decoder_layer_block.py
```
### hf_phi3 Blocks
#### Phi3 Model
```
python hf_qwen2/phi3_lora_model.py
```
### hf_mistral-nemo Blocks
#### Mistral-nemo Model
```
python hf_mistral-nemo/mistral-nemo_lora_model.py
```
