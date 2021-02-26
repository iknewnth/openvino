// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <inference_engine.hpp>
#include "infer_request_wrap.hpp"

class RemoteHelper
{
    class Impl;
    std::unique_ptr<Impl> _impl;
public:
    RemoteHelper();
    ~RemoteHelper();

    void Init();
    InferenceEngine::ExecutableNetwork Create(InferenceEngine::Core& ie, const std::string& graphPath);
    void UpdateRequestRemoteBlob(const InferenceEngine::ConstInputsDataMap& info,
        InferReqWrap::Ptr& request,
        void* data,
        size_t data_size);
    void GetWxH(size_t& width, size_t& height);
};
