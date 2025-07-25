#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "infinicore.h"
#include "logical_or_nvidia.cuh"
#include <cstdint>

namespace op::logical_or::nvidia {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t output_desc,
                                  std::vector<infiniopTensorDescriptor_t> input_descs) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = output_desc->dtype();

    const auto &a_desc = input_descs.at(0);
    const auto &b_desc = input_descs.at(1);
    const auto c_shape = output_desc->shape();
    const auto a_shape = a_desc->shape();
    const auto b_shape = b_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_I8);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, output_desc, input_descs)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    std::cout << "at calutate workspace: " << workspace << std::endl;
    std::cout << "at calutate workspace sieze: " << workspace_size << std::endl;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    switch (_dtype) {
    case INFINI_DTYPE_BOOL:
        return _device_info->calculate<256, cuda::LogicalOrOp, bool>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I8:
        return _device_info->calculate<256, cuda::LogicalOrOp, int8_t>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::logical_or::nvidia
