// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef USE_REMOTE_MEM
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>
#include <regex>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "utils.hpp"
#include "remote_helper.hpp"
#include "WorkloadContext.h"
#include "RemoteMemory.h"
#include "hddl2/hddl2_params.hpp"
#include "ie_compound_blob.h"

using namespace InferenceEngine;

class RemoteHelper::Impl {
    WorkloadID _workloadId = -1;
    HddlUnite::WorkloadContext::Ptr _context;
    RemoteContext::Ptr _contextPtr;

public:
    void Init(InferenceEngine::Core& ie) {
        _context = HddlUnite::createWorkloadContext();
        _context->setContext(_workloadId);
        auto ret = registerWorkloadContext(_context);
        if (ret != HddlStatusCode::HDDL_OK) {
            THROW_IE_EXCEPTION << "registerWorkloadContext failed with " << ret;
        }

        // init context map and create context based on it
        ParamMap paramMap = { {HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), _workloadId} };
        _contextPtr = ie.CreateContext("VPUX", paramMap);
    }

    HddlUnite::RemoteMemory::Ptr allocateRemoteMemory(const void* data, const size_t& dataSize) {
        auto remoteFrame = std::make_shared<HddlUnite::RemoteMemory>(*_context,
            HddlUnite::RemoteMemoryDesc(dataSize, 1, dataSize, 1));

        if (remoteFrame == nullptr) {
            THROW_IE_EXCEPTION << "Failed to allocate remote memory.";
        }

        if (remoteFrame->syncToDevice(data, dataSize) != HDDL_OK) {
            THROW_IE_EXCEPTION << "Failed to sync memory to device.";
        }
        return remoteFrame;
    }

    void PreallocRemoteMem(const InferenceEngine::ConstInputsDataMap& info,
        InferReqWrap::Ptr& request,
        const Blob::Ptr& inputBlob) {
        MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
        const TensorDesc& inputTensor = minput->getTensorDesc();
        // locked memory holder should be alive all time while access to its buffer happens
        auto minputHolder = minput->rmap();
        auto inputBlobData = minputHolder.as<uint8_t*>();

        // 1, allocate memory with HddlUnite on device
        auto remoteMemory = allocateRemoteMemory(inputBlobData, minput->byteSize());

        // 2, create remote blob by using already exists remote memory and specify color format of it
        ParamMap blobParamMap = { {HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory} };
        RemoteBlob::Ptr remoteBlobPtr = _contextPtr->CreateBlob(inputTensor, blobParamMap);
        if (remoteBlobPtr == nullptr) {
            THROW_IE_EXCEPTION << "CreateBlob failed.";
        }

        // 3, set remote blob
        request->setBlob(info.begin()->first, remoteBlobPtr);
    }

    RemoteContext::Ptr getRemoteContext() {
        return _contextPtr;
    }
};

RemoteHelper::RemoteHelper() : _impl(new RemoteHelper::Impl()) {
}

RemoteHelper::~RemoteHelper() {
}

void RemoteHelper::Init(InferenceEngine::Core& ie) {
    _impl->Init(ie);
}

void RemoteHelper::PreallocRemoteMem(const InferenceEngine::ConstInputsDataMap& info, InferReqWrap::Ptr& request, const Blob::Ptr& inputBlob) {
    _impl->PreallocRemoteMem(info, request, inputBlob);
}

InferenceEngine::RemoteContext::Ptr RemoteHelper::getRemoteContext() {
    return _impl->getRemoteContext();
}
#endif
