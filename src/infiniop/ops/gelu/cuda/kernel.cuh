#ifndef __GELU_CUDA_H__
#define __GELU_CUDA_H__

namespace op::gelu::cuda {

typedef struct GeluOp {
private:
    float pi = 3.1415927f;
    float kappa = 0.044715f;
    float beta = sqrtf(2 / pi);

    half2 h2_kappa = __float2half2_rn(kappa);
    half h_kappa = __float2half(kappa);
    cuda_bfloat162 b2_kappa = __float2bfloat162_rn(kappa);
    cuda_bfloat16 b_kappa = __float2bfloat16(kappa);

    half2 h2_beta = __float2half2_rn(beta);
    half h_beta = __float2half(beta);
    cuda_bfloat162 b2_beta = __float2bfloat162_rn(beta);
    cuda_bfloat16 b_beta = __float2bfloat16(beta);

    half2 h2_point_fiv = __float2half2_rn(0.5f);
    half h_point_fiv = __float2half(0.5f);
    cuda_bfloat162 b2_point_fiv = __float2bfloat162_rn(0.5f);
    cuda_bfloat16 b_point_fiv = __float2bfloat16(0.5f);

    half2 h2_one = __float2half2_rn(1.0f);
    half h_one = __float2half(1.0f);
    cuda_bfloat162 b2_one = __float2bfloat162_rn(1.0f);
    cuda_bfloat16 b_one = __float2bfloat16(1.0f);


public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        // use Approximate formula Gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.0044715 * x **3)))
        //  \kappa = 0.044715, \beta = sqrt(2 / pi)
        //  inner() = \beta * (x + \kappa * x ** 3)
        //  Gelu(x) = 0.5 * x * (1 + tanh(inner()))

        if constexpr (std::is_same_v<T, half2>) {
            half2 inner = __hmul2(h2_beta, __hadd2(input, __hmul2(h2_kappa, __hmul2(input, __hmul2(input, input)))));
            half2 res = __hmul2(h2_point_fiv, __hmul2(input, __hadd2(h2_one, h2tanh_approx(inner))));

            return res;

        } else if constexpr (std::is_same_v<T, half>) {
            half inner = __hmul(h_beta, __hadd(input, __hmul(h_kappa, __hmul(input, __hmul(input, input)))));
            half res = __hmul(h_point_fiv, __hmul(input, __hadd(h_one, htanh_approx(inner))));

            return res;

        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            cuda_bfloat162 inner = __hmul2(b2_beta, __hadd2(input, __hmul2(b2_kappa, __hmul2(input, __hmul2(input, input)))));
            cuda_bfloat162 res = __hmul2(b2_point_fiv, __hmul2(input, __hadd2(b2_one, h2tanh_approx(inner))));

            return res;

        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 inner = __hmul(b_beta, __hadd(input, __hmul(b_kappa, __hmul(input, __hmul(input, input)))));
            cuda_bfloat16 res = __hmul(b_point_fiv, __hmul(input, __hadd(b_one, htanh_approx(inner))));

            return res;

        } else if constexpr (std::is_same_v<T, float>) {
            float inner = __fmul_rn(beta, __fadd_rn(input, __fmul_rn(kappa, __fmul_rn(input, __fmul_rn(input, input)))));
            float res = __fmul_rn(0.5f, __fmul_rn(input, __fadd_rn(1.0f, tanhf(inner))));
            
            return res;
        } 
    };
} GeluOp;
} // namespace op::gelu::cuda

#endif // __silu_CUDA_H__
