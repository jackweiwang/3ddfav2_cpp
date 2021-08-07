#ifndef FACE3D_TNN_HPP
#define FACE3D_TNN_HPP


#include "tnn_sdk_sample.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

namespace TNN_NS {

typedef ObjectInfo Face3dInfo;
class Face3dInput : public TNNSDKInput {
public:
    Face3dInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~Face3dInput(){}
};

class Face3dOutput : public TNNSDKOutput {
public:
    Face3dOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~Face3dOutput() {};
    std::vector<Face3dInfo> face_list;

};

class Face3dOption : public TNNSDKOption {
public:
    Face3dOption() {}
    virtual ~Face3dOption() {}
    int input_width;
    int input_height;

    int num_thread = 1;
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

    void GenerateLandmarks(std::vector<Face3dInfo> &detects, Mat &landmarks,
                      int image_w, int image_h);

    int num_keypoints = 68;
};

}

#endif // TNN_EXAMPLES_BASE_BLAZEFACE_DETECTOR_H_
