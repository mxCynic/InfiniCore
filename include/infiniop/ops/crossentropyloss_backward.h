#ifndef __INFINIOP_CROSSENTROPYLOSSBACKWARD_API_H__
#define __INFINIOP_CROSSENTROPYLOSSBACKWARD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCrossEntropyLossBackWardDescriptor_t;

__C __export infiniStatus_t infiniopCreateCrossEntropyLossBackWardDescriptor(infiniopHandle_t handle,
                                                                             infiniopCrossEntropyLossBackWardDescriptor_t *desc_ptr,
                                                                             infiniopTensorDescriptor_t c,
                                                                             infiniopTensorDescriptor_t a,
                                                                             infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetCrossEntropyLossBackWardWorkspaceSize(infiniopCrossEntropyLossBackWardDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopCrossEntropyLossBackWard(infiniopCrossEntropyLossBackWardDescriptor_t desc,
                                                             void *workspace,
                                                             size_t workspace_size,
                                                             void *c,
                                                             const void *a,
                                                             const void *b,
                                                             void *stream);

__C __export infiniStatus_t infiniopDestroyCrossEntropyLossBackWardDescriptor(infiniopCrossEntropyLossBackWardDescriptor_t desc);

#endif
