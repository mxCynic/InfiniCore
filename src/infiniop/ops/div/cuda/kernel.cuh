#ifndef __DIV_CUDA_H__
#define __DIV_CUDA_H__

// #include "../../../devices/nvidia/nvidia_kernel_common.cuh"
namespace op::div::cuda {
typedef struct DivOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half>) {
            return __hdiv(a, b);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __hdiv(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return fdividef(a, b);
        } else if constexpr (std::is_same_v<T, double>) {
            return fdivide(a, b);
        } else {
            return a / b;
        }
    }
} DivOp;
typedef struct DivOpTrunc {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half>) {
            return htrunc(__hdiv(a, b));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return htrunc(__hdiv(a, b));
        } else if constexpr (std::is_same_v<T, float>) {
            return truncf(fdividef(a, b));
        } else if constexpr (std::is_same_v<T, double>) {
            return trunc(fdivide(a, b));
        } else {
            return a / b;
        }
    }
} DivOpTrunc;
typedef struct DivOpFloor {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half>) {
            return hfloor(__hdiv(a, b));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return hfloor(__hdiv(a, b));
        } else if constexpr (std::is_same_v<T, float>) {
            return floorf(fdividef(a, b));
        } else if constexpr (std::is_same_v<T, double>) {
            return floor(fdivide(a, b));
        } else {
            return a / b;
        }
    }
} DivOpFloor;
} // namespace op::div::cuda

#endif // __DIV_CUDA_H__
