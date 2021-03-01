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

class RemoteHelper {
    class Impl;
    std::unique_ptr<Impl> _impl;
public:
    RemoteHelper();
    ~RemoteHelper();

    void Init(InferenceEngine::Core& ie);
    void UpdateRequestRemoteBlob(const InferenceEngine::ConstInputsDataMap& info,
        InferReqWrap::Ptr& request,
        const InferenceEngine::Blob::Ptr& inputBlob);
    InferenceEngine::RemoteContext::Ptr getRemoteContext();
};
