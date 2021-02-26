// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef USE_PREALLOC_MEM
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
#include "hddl2_params.hpp"
#include "ie_compound_blob.h"

using namespace InferenceEngine;

#define REMOTE_IMAGE_WIDTH 1920
#define REMOTE_IMAGE_HEIGHT 1080

class RemoteHelper::Impl {
    WorkloadID _workloadId = -1;
    HddlUnite::WorkloadContext::Ptr _context;
    RemoteContext::Ptr _contextPtr;
    size_t _width = REMOTE_IMAGE_WIDTH;
    size_t _height = REMOTE_IMAGE_HEIGHT;
    std::string _modelPath;

public:
    void Init() {
        _context = HddlUnite::createWorkloadContext();
        _context->setContext(_workloadId);
        auto ret = registerWorkloadContext(_context);
        if (ret != HddlStatusCode::HDDL_OK) {
            THROW_IE_EXCEPTION << "registerWorkloadContext failed with " << ret;
        }
    }

    ExecutableNetwork Create(Core& ie, const std::string& graphPath) {
        _modelPath = graphPath;
        // init context map and create context based on it
        ParamMap paramMap = { {HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), _workloadId} };
        _contextPtr = ie.CreateContext("VPUX", paramMap);

        //
        // import network providing context as input to bind to context
        //
        std::filebuf blobFile;
        if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
            THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
        }
        std::istream graphBlob(&blobFile);
        return ie.ImportNetwork(graphBlob, _contextPtr);
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

    void UpdateRequestRemoteBlob(const InferenceEngine::ConstInputsDataMap& info,
        InferReqWrap::Ptr& request,
        void* data,
        size_t data_size) {
        // 1, allocate memory with HddlUnite on device
        auto remoteMemory = allocateRemoteMemory(data, data_size);

        // 2, create remote blob by using already exists remote memory and specify color format of it
        ParamMap blobParamMap = { {HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory},
            {HDDL2_PARAM_KEY(COLOR_FORMAT), ColorFormat::NV12} };

        TensorDesc inputTensor = TensorDesc(Precision::U8, { 1, 3, _height, _width }, Layout::NCHW);
        RemoteBlob::Ptr remoteBlobPtr = _contextPtr->CreateBlob(inputTensor, blobParamMap);
        if (remoteBlobPtr == nullptr) {
            THROW_IE_EXCEPTION << "CreateBlob failed.";
        }

        // 2.1, set preprocessing of remote blob
        const std::string inputName = info.begin()->first;
        PreProcessInfo preprocInfo = request->getPreProcess(inputName);
        preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
        preprocInfo.setColorFormat(ColorFormat::NV12);

        // 2.2, create roi blob
        ROI roiIE;
        roiIE = ROI{ 0, 0, 0, _width, _height };
        auto remoteBlobROIPtr = remoteBlobPtr->createROI(roiIE);

        // 3, set remote NV12 blob with preprocessing information
        request->setBlob(inputName, remoteBlobROIPtr, preprocInfo);
    }

    void GetWxH(size_t& width, size_t& height) {
        width = _width;
        height = _height;
    }
};

RemoteHelper::RemoteHelper() : _impl(new RemoteHelper::Impl()) {
}

RemoteHelper::~RemoteHelper() {
}

void RemoteHelper::Init() {
    _impl->Init();
}

InferenceEngine::ExecutableNetwork RemoteHelper::Create(InferenceEngine::Core& ie, const std::string& graphPath) {
    return _impl->Create(ie, graphPath);
}

void RemoteHelper::UpdateRequestRemoteBlob(const InferenceEngine::ConstInputsDataMap& info, InferReqWrap::Ptr& request, void* data, size_t data_size) {
    _impl->UpdateRequestRemoteBlob(info, request, data, data_size);
}

void RemoteHelper::GetWxH(size_t& width, size_t& height) {
    _impl->GetWxH(width, height);
}
#endif
