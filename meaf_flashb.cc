#include <vector>
#include "paddle/extension.h"



std::vector<paddle::Tensor> MeaFFlashBFwd(const Tensor& query, const Tensor& key, const Tensor& value, 
                            double dropout,bool causal, bool is_test){
    std::vector<paddle::Tensor> res;
    return res;
}

std::vector<paddle::Tensor> MeaFFlashBBwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, 
const Tensor& softmax_lse, const Tensor& seed_offset, const Tensor& out_grad, float dropout, bool causal){
    //  float dropout, bool causal, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad
    std::vector<paddle::Tensor> res;
    return res;
}

std::vector<paddle::DataType> MeaFFlashBFwdDtype(paddle::DataType q_dtype,
                                              paddle::DataType k_dtype,
                                              paddle::DataType v_dtype) {

  return {paddle::DataType::FLOAT32, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> MeaFFlashBFwdInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape,
    double dropout,
    bool causal, 
    bool is_test) {
  return {q_shape, k_shape, v_shape};
}

std::vector<std::vector<int64_t>> MeaFFlashBBwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    std::vector<int64_t> mean_shape,
    std::vector<int64_t> invvar_shape,
    std::vector<int64_t> dy_shape,
    float epsilon) {

    //seqstart_k, seqstart_q, max_seqlen_q, max_seqlen_k = None, None, -1, -1  
    //causal_diagonal = None
    //seqlen_k = None
    //scale = -1.0 
    //bias = None
    //PD_CHECK()
  return {x_shape, scale_shape, scale_shape};
}


PD_BUILD_OP(meaf_flashb)
    .Inputs({"q", "k", "v"})
    .Outputs({"out", "lse", "seed_offset"})
    .Attrs({"dropout: float, casual: bool, is_test: bool"})
    .SetKernelFn(PD_KERNEL(MeaFFlashBFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(MeaFFlashBFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MeaFFlashBFwdDtype));

PD_BUILD_GRAD_OP(meaf_flashb)
    .Inputs({"q", "k", "v", "out","lse","seed_offset",paddle::Grad("out")})
    .Outputs({paddle::Grad("q"), paddle::Grad("k"), paddle::Grad("v")})
    .Attrs({"dropout: float, casual: bool"})
    .SetKernelFn(PD_KERNEL(MeaFFlashBBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(MeaFFlashBBwdInferShape));


