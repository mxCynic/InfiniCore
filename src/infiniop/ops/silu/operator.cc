#include "../../handle.h"
#include "infinicore.h"
#include "infiniop/ops/silu.h"

#ifdef ENABLE_CPU_API
#include "cpu/silu_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/silu_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateSiluDescriptor(
    infiniopHandle_t handle,
    infiniopSiluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
#define CTEATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::silu::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::silu::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                         \
            {x_desc})

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CTEATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CTEATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CTEATE
}

__C infiniStatus_t infiniopGetSiluWorkspaceSize(infiniopSiluDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                \
    case CASE:                                                                              \
        *size = reinterpret_cast<op::silu::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopSilu(
    infiniopSiluDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::silu::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, {x}, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__C infiniStatus_t infiniopDestroySiluDescriptor(infiniopSiluDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::silu::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}
