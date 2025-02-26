import numpy as np
import random
import torch
            
def dummy_packed_attn_mask(batch_size, min_len, seq_len):
    assert batch_size > 0, f"{batch_size} should be greater than 0!"
    assert seq_len > 0, f"{seq_len} needs to be greater than zero"
    max_len = seq_len - min_len + 1
    assert max_len > 0, f"{seq_len} - {min_len} needs to be greater than zero"
    attn_mask = []
    position_ids = []
    for batch in range(batch_size):
        idx = 0
        mask = []
        positions = []
        while idx < max_len:
            # Range range requires an exclusive end range.
            rand_len = random.randrange(min_len, seq_len - idx + 1)
            seq_mask = ([0] * idx) + ([1] * rand_len) + ([0] * (seq_len - (idx + rand_len)))
            positions += range(rand_len)
            #print("batch_size", batch_size, "min_len", min_len, "seq_len", seq_len, "batch", batch, "idx", idx, "rand_len", rand_len)
            #print("seq_mask", seq_mask)
            for _ in range(rand_len):
                mask.append(seq_mask)
            idx += rand_len
        # add rows of zeros if necessary
        if idx < seq_len:
            mask.append([0] * seq_len)
            positions.append(0)
        attn_mask.append(mask)
        position_ids.append(positions)
    attn_mask_tensor = torch.tensor(attn_mask, device='cuda', dtype=torch.uint8)
    position_ids_tensor = torch.tensor(position_ids, device='cuda', dtype=torch.long)
    return attn_mask_tensor, position_ids_tensor

if __name__ == "__main__":
    mask, pos_ids = dummy_packed_attn_mask(2, 2, 16)
    print(mask)
    print(pos_ids)
