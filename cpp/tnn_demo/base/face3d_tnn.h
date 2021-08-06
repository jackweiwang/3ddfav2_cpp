#ifndef FACE3D_TNN_HPP
#define FACE3D_TNN_HPP


#include "tnn_sdk_sample.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

namespace TNN_NS {

//typedef ObjectInfo BlazeFaceInfo;

class Face3dInput : public TNNSDKInput {
public:
    Face3dInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~Face3dInput(){}
};

class Face3dOutput : public TNNSDKOutput {
public:
    Face3dOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~Face3dOutput() {};
    std::vector<float > face_list;
};

class Face3dOption : public TNNSDKOption {
public:
    Face3dOption() {}
    virtual ~Face3dOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    float min_score_threshold = 0.75;
    float min_suppression_threshold = 0.3;
    std::string anchor_path;
};

class Face3d : public TNNSDKSample {
public:
    virtual ~Face3d() {};
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                            std::string name = kTNNSDKDefaultName);
    
private:

    
    std::vector<float> anchors;
    
    int num_anchors = 896;
    int detect_dims = 16;
    int num_keypoints = 6;
};

}

#endif // TNN_EXAMPLES_BASE_BLAZEFACE_DETECTOR_H_
// class Face3d
// {
//     public:
//         Face3d(const std::string& model_path);
//         ~Face3d();

//         std::vector<float> Predict(const uint8_t* buffer_120x120x3);

//     private:
//         std::unique_ptr<MNN::CV::ImageProcess> _preprocess;
//         std::unique_ptr<MNN::Interpreter> _interpreter;
//         MNN::Session* _session;
// };
// #endif