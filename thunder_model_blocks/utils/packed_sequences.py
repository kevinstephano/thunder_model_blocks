import random
            
def gen_attn_mask(batch_size, min_len, seq_len):
    assert batch_size > 0, f"{batch_size} should be greater than 0!"
    max_len = seq_len - min_len
    assert max_len > 0, f"{seq_len} - {min_len} needs to be greater than zero"
    attn_mask = []
    for _ in cfg.batch_size:
        idx = 0
        while idx < max_len:
            rand_len = random.randrange(min_len, max_len - idx + 1)
            seq_mask = ([0] * idx) + ([1] * rand_len) + ([1] * (seq_len - (idx + rand_len))
            for _ in range(rand_len):
                attn_mask.append(seq_mask)
    return attn_mask
