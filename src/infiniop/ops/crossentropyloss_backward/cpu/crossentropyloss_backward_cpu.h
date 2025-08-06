#ifndef __CrossEntropyLossBackWard_CPU_H__
#define __CrossEntropyLossBackWard_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(CrossEntropyLossBackWard, cpu)

namespace op::CrossEntropyLossBackWard::cpu {
typedef struct CrossEntropyLossBackWardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b, const size_t N) const {
        return (a - b) / N;
    }
} CrossEntropyLossBackWardOp;
} // namespace op::CrossEntropyLossBackWard::cpu

#endif // __CrossEntropyLossBackWard_CPU_H__
