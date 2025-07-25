#ifndef __INFINIOP_SILU_API_H__
#define __INFINIOP_SILU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSiluDescriptor_t;

__C __export infiniStatus_t infiniopCreateSiluDescriptor(infiniopHandle_t handle,
                                                         infiniopSiluDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y,
                                                         infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetSiluWorkspaceSize(infiniopSiluDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopSilu(infiniopSiluDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *y,
                                         const void *x,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroySiluDescriptor(infiniopSiluDescriptor_t desc);

#endif
