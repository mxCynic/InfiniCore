#include "../../handle.h"
#include "infinicore.h"
#include "infiniop/ops/logical_or.h"

#ifdef ENABLE_CPU_API
#include "cpu/logical_or_cpu.h"
#endif
#ifdef ENABLE_NVIDIA_API
#include "nvidia/logical_or_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateLogicalOrDescriptor(
    infiniopHandle_t handle,
    infiniopLogicalOrDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
#define CTEATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::logical_or::NAMESPACE::Descriptor::create(                     \
            handle,                                                               \
            reinterpret_cast<op::logical_or::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                               \
            {a_desc, b_desc})

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

__C infiniStatus_t infiniopGetLogicalOrWorkspaceSize(infiniopLogicalOrDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<op::logical_or::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__C infiniStatus_t infiniopLogicalOr(
    infiniopLogicalOrDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                       \
        return reinterpret_cast<const op::logical_or::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, c, {a, b}, stream)

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

__C infiniStatus_t infiniopDestroyLogicalOrDescriptor(infiniopLogicalOrDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        delete reinterpret_cast<const op::logical_or::NAMESPACE::Descriptor *>(desc); \
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
