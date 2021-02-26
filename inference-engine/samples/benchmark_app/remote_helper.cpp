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
#include "../../src/plugin_api/blob_factory.hpp"

using namespace InferenceEngine;

#define REMOTE_IMAGE_WIDTH 224
#define REMOTE_IMAGE_HEIGHT 224

class RemoteHelper::Impl
{
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

    // xxx
    template<typename blobData_t>
    std::vector<size_t> yieldTopClasses(const InferenceEngine::Blob::Ptr& resultBlob, size_t maxClasses) {
        const blobData_t* unsortedRawData = resultBlob->cbuffer().as<const blobData_t*>();
        // map key is a byte from raw data (quantized probability)
        // map value is the index of that byte (class id)
        std::multimap<blobData_t, size_t> sortedClassMap;
        for (size_t classIndex = 0; classIndex < resultBlob->size(); classIndex++) {
            blobData_t classProbability = unsortedRawData[classIndex];
            std::pair<blobData_t, size_t> mapItem(classProbability, classIndex);
            sortedClassMap.insert(mapItem);
        }

        std::vector<size_t> topClasses;
        for (size_t classCounter = 0; classCounter < maxClasses; classCounter++) {
            typename std::multimap<blobData_t, size_t>::reverse_iterator classIter = sortedClassMap.rbegin();
            std::advance(classIter, classCounter);
            topClasses.push_back(classIter->second);
        }

        return topClasses;
    }
    void Test(InferReqWrap::Ptr& request, std::string modelPath,
        void* data, size_t data_size, PreProcessInfo* preprocInfo) {

        std::string outputBlobName;

        // --- Reference Blob
        //Blob::Ptr refBlob = ReferenceHelper::CalcCpuReferenceSingleOutput(modelPath, inputNV12Blob, preprocInfo);
        Blob::Ptr refBlob;
        {
            Blob::Ptr inputBlob;// = make_blob_with_precision(rgbFrameTensor);
            //inputBlob->allocate();
            {
                auto yPlane = InferenceEngine::make_shared_blob<uint8_t>(
                    TensorDesc(Precision::U8, { 1, 1, _height, _width }, Layout::NHWC),
                    (uint8_t*)data);

                auto uvPlane = InferenceEngine::make_shared_blob<uint8_t>(
                    TensorDesc(Precision::U8, { 1, 2, _height / 2, _width / 2 }, Layout::NHWC),
                    (uint8_t*)data + _width * _height);

                inputBlob = InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(yPlane, uvPlane);
            }
            //{
            //    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
            //    // locked memory holder should be alive all time while access to its buffer happens
            //    auto minputHolder = minput->wmap();
            //    auto inputBlobData = minputHolder.as<uint8_t*>();
            //    memcpy(inputBlobData, data, data_size);
            //}

            std::cout << "Calculating reference on CPU (single output)..." << std::endl;
            Core ie;
            auto network = ie.ReadNetwork(modelPath);

            OutputsDataMap outputs_info = network.getOutputsInfo();
            const size_t NUM_OUTPUTS = 1;
            if (outputs_info.size() != NUM_OUTPUTS) {
                THROW_IE_EXCEPTION << "Number of outputs isn't equal to 1";
            }
            outputBlobName = outputs_info.begin()->first;

            InputsDataMap input_info = network.getInputsInfo();
            for (auto& item : input_info) {
                auto input_data = item.second;
                input_data->setPrecision(Precision::U8);
            }

            ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = executableNetwork.CreateInferRequest();

            auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
            if (preprocInfo != nullptr) {
                inferRequest.SetBlob(inputBlobName, inputBlob, *preprocInfo);
            }
            else {
                inferRequest.SetBlob(inputBlobName, inputBlob);
            }

            ConstOutputsDataMap output_info = executableNetwork.GetOutputsInfo();
            for (const auto& output : output_info) {
                const auto outputBlobName = output.first;
                auto output_blob = make_blob_with_precision(output.second->getTensorDesc());
                output_blob->allocate();
                inferRequest.SetBlob(outputBlobName, output_blob);
            }

            inferRequest.Infer();

            refBlob = inferRequest.GetBlob(outputBlobName);
        }
        // test result
        request->infer();
        // --- Get output
        auto outputBlob = request->getBlob(outputBlobName);

        //Comparators::compareTopClassesUnordered(
        //    outputBlob, refBlob, 4);
        {
            size_t maxClasses = 14;
            std::vector<size_t> outTopClasses, refTopClasses;
            switch (outputBlob->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::U8:
                outTopClasses = yieldTopClasses<uint8_t>(outputBlob, maxClasses);
                refTopClasses = yieldTopClasses<uint8_t>(refBlob, maxClasses);
                break;
            case InferenceEngine::Precision::FP32:
                outTopClasses = yieldTopClasses<float>(outputBlob, maxClasses);
                refTopClasses = yieldTopClasses<float>(refBlob, maxClasses);
                break;
            default:
                throw std::runtime_error("compareTopClasses: only U8 and FP32 are supported");
            }
            std::ostringstream logStream;
            logStream << "out: ";
            for (size_t classId = 0; classId < maxClasses; classId++) {
                logStream << outTopClasses[classId] << " ";
            }

            logStream << std::endl << "ref: ";
            for (size_t classId = 0; classId < maxClasses; classId++) {
                logStream << refTopClasses[classId] << " ";
            }
            auto ret = std::is_permutation(outTopClasses.begin(), outTopClasses.end(),
                refTopClasses.begin());
            if(!ret){
                std::cout << "error result: " << logStream.str() << std::endl;
            }
        }
    }
    // xxx end

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

        // xxx
        Test(request, R"(D:\lc\Work\kmb-plugin\temp\models\src\models\KMB_models\INT8\public\ResNet-50\resnet50_uint8_int8_weights_pertensor.xml)", data, data_size, &preprocInfo);
        // xxx end
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
