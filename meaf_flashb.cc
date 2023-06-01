#include "paddle/extension.h"
#include <vector>

using paddle::Tensor;

namespace paddle {
namespace experimental {
PADDLE_API void flash_attn_grad(const Tensor &q, const Tensor &k,
                                const Tensor &v, const Tensor &out,
                                const Tensor &softmax_lse,
                                const Tensor &seed_offset,
                                const Tensor &out_grad, float dropout,
                                bool causal, Tensor *q_grad, Tensor *k_grad,
                                Tensor *v_grad);
}
} // namespace paddle

std::vector<Tensor> MeaFFlashBFwd(const Tensor &q, const Tensor &k,
                                  const Tensor &v, bool causal, bool is_test) {
  // not used value
  paddle::optional<Tensor> bias;
  paddle::optional<Tensor> cu_seqlens_q;
  paddle::optional<Tensor> cu_seqlens_k;
  paddle::optional<Tensor> causal_diagonal;
  paddle::optional<Tensor> seqlen_k;
  paddle::Scalar max_seqlen_q = -1;
  paddle::Scalar max_seqlen_k = -1;

  std::vector<Tensor> res(3);
  std::tie(res[0], res[1], res[2]) =
      paddle::experimental::memory_efficient_attention(
          q, k, v, bias, cu_seqlens_q, cu_seqlens_k, causal_diagonal,
          cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, 0.0, -1.0, is_test);
  // b s h hidden
  auto q_shape = q.shape();
  // b h s
  auto logsumexp_shape = res[1].shape();
  PD_CHECK(q_shape.size() == 4);
  PD_CHECK(logsumexp_shape.size() == 3);
  PD_CHECK(q_shape[0] == logsumexp_shape[0]);
  PD_CHECK(q_shape[1] == logsumexp_shape[2]);
  PD_CHECK(q_shape[2] == logsumexp_shape[1]);
  return res;
}

std::vector<paddle::Tensor> MeaFFlashBBwd(const Tensor &q, const Tensor &k,
                                          const Tensor &v, const Tensor &out,
                                          const Tensor &softmax_lse,
                                          const Tensor &seed_offset,
                                          const Tensor &out_grad, bool causal) {
  std::vector<paddle::Tensor> res(3);
  paddle::experimental::flash_attn_grad(q, k, v, out, softmax_lse, seed_offset,
                                        out_grad, 0.0, causal, &res[0], &res[1],
                                        &res[2]);
  return res;
}

std::vector<paddle::DataType> MeaFFlashBFwdDtype(paddle::DataType q_dtype,
                                                 paddle::DataType k_dtype,
                                                 paddle::DataType v_dtype) {
  return {q_dtype, paddle::DataType::FLOAT32, paddle::DataType::INT64};
}

std::vector<std::vector<int64_t>> MeaFFlashBFwdInferShape(
    std::vector<int64_t> q_shape, std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape, bool causal, bool is_test) {

  PD_CHECK(q_shape.size() == 4, "q_shape.size() returns ", q_shape.size(),
           ", expected 4.");
  PD_CHECK(k_shape.size() == 4, "k_shape.size() returns ", k_shape.size(),
           ", expected 4.");
  PD_CHECK(v_shape.size() == 4, "v_shape.size() returns ", v_shape.size(),
           ", expected 4.");

  const int64_t query_batch_size = q_shape[0];
  const int64_t query_seq_length = q_shape[1];
  const int64_t query_num_head = q_shape[2];
  const int64_t query_head_size = q_shape[3];

  const int64_t key_batch_size = k_shape[0];
  const int64_t key_seq_length = k_shape[1];
  const int64_t key_num_head = k_shape[2];
  const int64_t key_head_size = k_shape[3];

  const int64_t value_batch_size = v_shape[0];
  const int64_t value_seq_length = v_shape[1];
  const int64_t value_num_head = v_shape[2];
  const int64_t value_head_size = v_shape[3];

  PD_CHECK(query_batch_size == key_batch_size);
  PD_CHECK(key_batch_size == value_batch_size);
  PD_CHECK(query_num_head == key_num_head);
  PD_CHECK(key_num_head == value_num_head);
  PD_CHECK(query_head_size == key_head_size);
  PD_CHECK(key_seq_length == value_seq_length);

  std::vector<int64_t> out_dims(
      {query_batch_size, query_seq_length, query_num_head, value_head_size});
  std::vector<int64_t> logsumexp_dims(
      {query_batch_size, query_num_head, query_seq_length});
  std::vector<int64_t> seed_and_offset_dims({2});
  return {out_dims, logsumexp_dims, seed_and_offset_dims};
}

std::vector<std::vector<int64_t>> MeaFFlashBBwdInferShape(
    std::vector<int64_t> q_shape, std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape, std::vector<int64_t> out_shape,
    std::vector<int64_t> softmax_lse_shape,
    std::vector<int64_t> seed_offset_shape, std::vector<int64_t> out_grad_shape,
    bool causal) {

  return {q_shape, k_shape, v_shape};
}

PD_BUILD_OP(meaf_flashb)
    .Inputs({"q", "k", "v"})
    .Outputs({"out", "lse", "seed_offset"})
    .Attrs({"casual: bool", "is_test: bool"})
    .SetKernelFn(PD_KERNEL(MeaFFlashBFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(MeaFFlashBFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MeaFFlashBFwdDtype));

PD_BUILD_GRAD_OP(meaf_flashb)
    .Inputs({"q", "k", "v", "out", "lse", "seed_offset", paddle::Grad("out")})
    .Outputs({paddle::Grad("q"), paddle::Grad("k"), paddle::Grad("v")})
    .Attrs({"casual: bool"})
    .SetKernelFn(PD_KERNEL(MeaFFlashBBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(MeaFFlashBBwdInferShape));
