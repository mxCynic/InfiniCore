
#ifndef __SILU_CUDA_H__
#define __SILU_CUDA_H__

namespace op::silu::cuda {
typedef struct SiluOp {
private:
    template <typename T>
    T sigmoid(const T &x) const {
        return T(1) / (T(1) + std::exp(-x));
    }

public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        return x * sigmoid(x);
    }
} SiluOp;
} // namespace op::silu::cuda

#endif // __silu_CUDA_H__
