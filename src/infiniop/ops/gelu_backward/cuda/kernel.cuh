#ifndef __gelu_backward_CUDA_H__
#define __gelu_backward_CUDA_H__

namespace op::gelu_backward::cuda {
typedef struct GeluBackWardOp {
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

    half2 h2_three = __float2half2_rn(3.0f);
    half h_three = __float2half(3.0f);
    cuda_bfloat162 b2_three = __float2bfloat162_rn(3.0f);
    cuda_bfloat16 b_three = __float2bfloat16(3.0f);

public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &input, const T &grad_output) const {
        // use Approximate formula GeluBackWard(x) = 0.5 [(1 + tanh(inner())) + x(1 - tanh(inner) ** 2)(d(inner()))]
        // use Approximate formula Gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.0044715 * x ** 3 )))
        //  \kappa = 0.044715, \beta = sqrt(2 / pi)
        //  inner() = \beta * (x + \kappa * x ** 3)
        //  dinner() = \beta * (1 + 3 * \kappa * x ** 2)
        //  Gelu(x) = 0.5 * x * (1 + tanh(inner()))
        //  d(Gelu(x)) = 0.5 [(1 + tanh(inner())) + x(1 - tanh(inner) ** 2)(d(inner()))]
        //
        //  GeluBackWard(x) = grad_out  * d(Gelu(x))

        if constexpr (std::is_same_v<T, half2>) {
            // half2 inner = h2_beta * (input +  h2_kappa *  input * input * input);
            half2 inner = __hmul2(h2_beta, __hadd2(input, __hmul2(h2_kappa, __hmul2(input, __hmul2(input, input)))));
            half2 dinner = __hmul2(h2_beta, __hadd2(h2_one, __hmul2(h2_three, __hmul2(h2_kappa, __hmul2(input, input)))));
            half2 tanh = h2tanh_approx(inner);
            half2 dGelu = __hmul2(h2_point_fiv, __hadd2(h2_one, __hadd2(tanh, __hmul2(input, __hmul2(__hadd2(h2_one, __hneg2(__hmul2(tanh, tanh))), dinner)))));

            return grad_output * dGelu;

        } else if constexpr (std::is_same_v<T, half>) {
            half inner = __hmul(h_beta, __hadd(input, __hmul(h_kappa, __hmul(input, __hmul(input, input)))));
            half dinner = __hmul(h_beta, __hadd(h_one, __hmul(h_three, __hmul(h_kappa, __hmul(input, input)))));
            half tanh = htanh_approx(inner);
            half dGelu = __hmul(h_point_fiv, __hadd(h_one, __hadd(tanh, __hmul(input, __hmul(__hadd(h_one, __hneg(__hmul(tanh, tanh))), dinner)))));

            return grad_output * dGelu;

        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            cuda_bfloat162 inner = __hmul2(b2_beta, __hadd2(input, __hmul2(b2_kappa, __hmul2(input, __hmul2(input, input)))));
            cuda_bfloat162 dinner = __hmul2(b2_beta, __hadd2(b2_one, __hmul2(b2_three, __hmul2(b2_kappa, __hmul2(input, input)))));
            cuda_bfloat162 tanh = h2tanh(inner);
            cuda_bfloat162 dGelu = __hmul2(b2_point_fiv, __hadd2(b2_one, __hadd2(tanh, __hmul2(input, __hmul2(__hadd2(b2_one, __hneg2(__hmul2(tanh, tanh))), dinner)))));

            return grad_output * dGelu;

        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 inner = __hmul(b_beta, __hadd(input, __hmul(b_kappa, __hmul(input, __hmul(input, input)))));
            cuda_bfloat16 dinner = __hmul(b_beta, __hadd(b_one, __hmul(b_three, __hmul(b_kappa, __hmul(input, input)))));
            cuda_bfloat16 tanh = htanh(inner);
            cuda_bfloat16 dGelu = __hmul(b_point_fiv, __hadd(b_one, __hadd(tanh, __hmul(input, __hmul(__hadd(b_one, __hneg(__hmul(tanh, tanh))), dinner)))));

            return grad_output * dGelu;
        } else if constexpr (std::is_same_v<T, float>) {
            float inner = __fmul_rn(beta, __fadd_rn(input, __fmul_rn(kappa, __fmul_rn(input, __fmul_rn(input, input)))));
            float dinner = __fmul_rn(beta, __fadd_rn(1.0f, __fmul_rn(3.0f, __fmul_rn(kappa, __fmul_rn(input, input)))));
            float tanh = tanhf(inner);
            float dGelu = __fmul_rn(0.5f, __fadd_rn(1.0f, __fadd_rn(tanh, __fmul_rn(input, __fmul_rn(__fsub_rn(1.0f, __fmul_rn(tanh, tanh)), dinner)))));

            return grad_output * dGelu;
        }
    };
} GeluBackWardOp;
} // namespace op::gelu_backward::cuda

#endif // __silu_CUDA_H__
