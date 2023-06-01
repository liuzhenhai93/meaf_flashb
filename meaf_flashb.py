import numpy as np
import paddle
from paddle import _C_ops
from paddle.incubate.nn.memory_efficient_attention import LowerTriangularMask

import paddle
from custom_attention import meaf_flashb

def flash_attn(query, key, value, dropout=0.0, causal=True):
    out, _, lse, seed_offset = _C_ops.flash_attn(query, key, value,
        dropout, causal, False, False)
    return out, lse, seed_offset

def custum_attention(query, key, value, dropout=0.0, causal=True):
    out, lse, seed_offset = meaf_flashb(query, key, value, dropout, causal, False)
    return out, lse, seed_offset

def mea(query, key, value, dropout=0.0, causal=True):
    seqstart_k, seqstart_q, max_seqlen_q, max_seqlen_k = None, None, -1, -1  
    causal_diagonal = None
    seqlen_k = None
    scale = -1.0 
    bias = None
    out, lse, seed_offset = _C_ops.memory_efficient_attention(
            query,
            key,
            value,
            bias,
            seqstart_q,
            seqstart_k,
            causal_diagonal,
            seqlen_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            dropout,
            scale,
            False,
        ) 
    return out, lse, seed_offset


shape = [1, 32640, 1, 128]
dtype = paddle.bfloat16
seed = 1000

query = paddle.randn(shape, dtype=dtype)
key = paddle.randn(shape, dtype=dtype)
value = paddle.randn(shape, dtype=dtype)

paddle.seed(seed)

offsets = []
for i in range(10): 
    out1, lse1, so1 = custum_attention(query, key, value)
    out2, lse2, so2 = mea(query, key, value)     
    so1 = so1.numpy().tolist()
    so2 = so2.numpy().tolist()
    assert len(so1) == 2
    assert len(so2) == 2
    assert so1[0] == so2[0]
    assert so1[0] == seed
    offsets.append(so1[1])
    offsets.append(so2[1])

interval = offsets[1] - offsets[0]
assert offsets == list(range(0, interval * len(offsets), interval)) 
print(offsets)


def max_diff(out1, out2, name):
    assert out1.shape == out2.shape
    out1 = out1.flatten()
    out2 = out2.flatten()
    diff = paddle.abs(out1 - out2)
    diff = diff.astype(paddle.float32).numpy()
    idx = np.argmax(diff)
    max_diff = diff[idx]
    mean_diff = np.mean(diff)
    max_diff_x = out1[idx].astype(paddle.float32).numpy()[0]
    max_diff_y = out2[idx].astype(paddle.float32).numpy()[0]
    print(f'FA VS MEA {name}: max diff {max_diff} ({max_diff_x} VS {max_diff_y}), mean diff {mean_diff}')

max_diff(out1, out2, "out")
max_diff(lse1, lse2, "log_sum_exp")
#print(dir(out1))