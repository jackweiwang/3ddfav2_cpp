// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef TNN_EXAMPLES_BASE_FACE_DETECT_MESH_H_
#define TNN_EXAMPLES_BASE_FACE_DETECT_MESH_H_

#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS{
typedef ObjectInfo FaceDetect3dInfo;
class FaceDetect3dInput : public TNNSDKInput {
public:
    FaceDetect3dInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~FaceDetect3dInput(){}
};

class FaceDetect3dOutput : public TNNSDKOutput {
public:
    FaceDetect3dOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~FaceDetect3dOutput() {};
    std::vector<FaceDetect3dInfo> face_list;

};

class FaceDetect3dOption : public TNNSDKOption {
public:
    FaceDetect3dOption() {}
    virtual ~FaceDetect3dOption() {}
    int input_width;
    int input_height;

    int num_thread = 1;
};
class FaceDetect3D : public TNN_NS::TNNSDKComposeSample {
public:
    virtual ~FaceDetect3D() {}

    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);

    virtual Status Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks);

protected:
    bool has_prev_face_ = false;

    std::shared_ptr<TNNSDKSample> predictor_detect_ = nullptr;
    std::shared_ptr<TNNSDKSample> predictor_3d_ = nullptr;
};

}

#endif // TNN_EXAMPLES_BASE_FACE_DETECT_MESH_H_
