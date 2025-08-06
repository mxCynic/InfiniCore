#ifndef __CROSSENTROPYLOSSBACKWARD_CUDA_H__
#define __CROSSENTROPYLOSSBACKWARD_CUDA_H__

namespace op::CrossEntropyLossBackWard::cuda {
typedef struct CrossEntropyLossBackWardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return __hadd(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fadd_rd(a, b);
        } else {
            return a + b;
        }
    }
} CrossEntropyLossBackWardOp;
} // namespace op::CrossEntropyLossBackWard::cuda

#endif // __CrossEntropyLossBackWard_CUDA_H__
