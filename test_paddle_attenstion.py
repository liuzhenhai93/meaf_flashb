import paddle
import paddle.nn.functional as F
import paddle.incubate.nn.attn_bias as ab
import numpy as np
import time

from paddle.nn.functional.flash_attention import (
    flash_attention,
    flash_attn_unpadded,
)
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
)

from custom_attention import meaf_flashb


class PaddleAttension(object):
    def __init__(self):
        pass

    def attention(self, q, k, v, causal, method = "naive"):
        if method == "naive":
            return self._naive_attention(q, k, v, causal)
        elif method == "flash":
            return self._flash_attention(q, k, v, causal)
        elif method == "mea":
            return self._memory_efficient_attention(q, k, v, causal)
        else:
            return self._meaf_flashb(q, k, v, causal)


    def _naive_attention(self, q, k, v, causal):
        # (b, seq, head, hidden) -> (b, head, seq, hidden)
        qt = paddle.transpose(q, [0, 2, 1, 3])
        kt = paddle.transpose(k, [0, 2, 1, 3])
        vt = paddle.transpose(v, [0, 2, 1, 3])
        # scale
        scale = 1.0 / np.sqrt(q.shape[-1])
        # q * k^t, (b, head, seq, hidden), (b, head, hidden, seq)-> (b, head, seq, seq)
        s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
        s = paddle.scale(s, scale)
        # mask or not
        p = (
            paddle.incubate.softmax_mask_fuse_upper_triangle(s)
            if causal
            else F.softmax(s)
        )
        # attention , (b, head, seq, seq) , (b, head, seq, hidden) -> (b, head, seq, hidden)
        o = paddle.matmul(p, vt)
        # (b, seq, head, hidden)
        return paddle.transpose(o, [0, 2, 1, 3])

    def _flash_attention(self, q, k, v, causal):
        out, _ = flash_attention(q, k, v, causal=causal)
        return out

    def _memory_efficient_attention(self, q, k, v, causal):
        scale = 1.0 / np.sqrt(q.shape[-1])
        att_bias = ab.LowerTriangularMask() if causal else None
        out = memory_efficient_attention(
            q,
            k,
            v,
            att_bias,
            0.0,
            scale,
            True
        )
        return out

    def _meaf_flashb(self, q, k, v, causal):
        out, lse, seed_offset = meaf_flashb(q, k, v, 0.0, causal, False)
        return out



def time_attention_func(attention, q, k, v, sync=False, iteration=10):
    forward_time = 0.0
    backward_time = 0.0
    for i in range(iteration):
        t = attention(q, k, v, True)
        t.backward()
        paddle.device.cuda.synchronize()
        begin = time.time()
        t2 = attention(q, k, v, True)
        if sync:
            paddle.device.cuda.synchronize()
        end = time.time()
        forward_time += (end - begin)
        t2.backward()
        paddle.device.cuda.synchronize()
        backward_time += (time.time() - begin)
    return forward_time / iteration ,  backward_time / iteration

def create_data(shape, dtype):
    paddle.seed(100)
    np.random.seed(0)
    q = np.random.random(shape)
    k = np.random.random(shape)
    v = np.random.random(shape)

    place = paddle.CUDAPlace(0)
    q = paddle.to_tensor(
            q, place=place, dtype=dtype, stop_gradient=False
        )
    k = paddle.to_tensor(
            k, place=place, dtype=dtype, stop_gradient=False
        )

    v = paddle.to_tensor(
            v, place=place, dtype=dtype, stop_gradient=False
        )
    return q, k, v    


def get_attention(paddle_attention, method):

    def attention1(q, k, v, causal):
        return paddle_attention._flash_attention(q, k, v, causal)

    def attention2(q, k, v, causal):
        return paddle_attention._memory_efficient_attention(q, k, v, causal)

    def attention3(q, k, v, causal):
        return paddle_attention._meaf_flashb(q, k, v, causal)

    if method == "flash" :
        return attention1
    elif method == "mea":
        return attention2
    else:
        return attention3        


def test_attention_precision():
    paddle_attention = PaddleAttension()
    shape = (1, 32*1024, 12, 128)
    causal = True
    dtype = 'bfloat16'

    inputs = [create_data(shape, dtype) for i in range(3)]
    att_types = ["mea", "flash", "meaf_flashb"]
    outputs = []
    paddle.device.cuda.synchronize()
    for (att, data) in zip(att_types, inputs):
        attention = get_attention(paddle_attention, att)
        t = attention(*data,True)
        t.backward()
        outputs.append(t)

    paddle.device.cuda.synchronize()    
    # check diff 

    def check_diff(context, a, b):
        assert a.shape == b.shape
        a = a.flatten()
        b = b.flatten()
        diff = paddle.abs(a - b)
        diff = diff.astype(paddle.float32).numpy()
        idx = np.argmax(diff)
        max_diff = diff[idx]
        mean_diff = np.mean(diff)
        max_diff_x = a[idx].astype(paddle.float32).numpy()[0]
        max_diff_y = b[idx].astype(paddle.float32).numpy()[0]
        print(f'{context}: max diff {max_diff} ({max_diff_x} VS {max_diff_y}), mean diff {mean_diff}')

    check_diff("mea vs flash out", outputs[0], outputs[1])
    check_diff("mea vs flash q-grad", inputs[0][0].grad, inputs[1][0].grad)     
    check_diff("mea vs flash k-grad", inputs[0][1].grad, inputs[1][1].grad)
    check_diff("mea vs flash v-grad", inputs[0][2].grad, inputs[1][2].grad)


    check_diff("mea vs meaf_flashb out", outputs[0], outputs[2])
    check_diff("mea vs meaf_flashb q-grad", inputs[0][0].grad, inputs[2][0].grad)     
    check_diff("mea vs meaf_flashb k-grad", inputs[0][1].grad, inputs[2][1].grad)
    check_diff("mea vs meaf_flashb v-grad", inputs[0][2].grad, inputs[2][2].grad)



def test_attention_perfromance():
    paddle_attention = PaddleAttension()
    shape = (1, 32*1024, 1, 128)
    causal = True
    dtype = 'bfloat16'
    q, k, v = create_data(shape, dtype)   
    paddle.device.cuda.synchronize()
    for att in ["mea", "flash", "meaf_flashb"]:
        attention = get_attention(paddle_attention, att)
        forward_time, backward_time = time_attention_func(attention, q, k, v)
        total_time = forward_time + backward_time
        forward_time, backward_time = time_attention_func(attention, q, k, v, True)
        backward_time_nosync = total_time - forward_time
        print(f"paddle-{att},x,1.0,1.0,{forward_time},{backward_time},{backward_time_nosync},{forward_time+backward_time}, {total_time}")

if __name__ =='__main__':
    test_attention_perfromance()
    test_attention_precision()
