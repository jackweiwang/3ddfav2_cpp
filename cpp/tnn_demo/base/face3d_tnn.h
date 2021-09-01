#ifndef FACE3D_TNN_HPP
#define FACE3D_TNN_HPP


#include "tnn_sdk_sample.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include "sample_timer.h"
#include "stdlib.h"
#include <algorithm>
#include <cstring>
#include <memory>
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
    Face3dInfo face;

};

class Face3dOption : public TNNSDKOption {
public:
    Face3dOption() {}
    virtual ~Face3dOption() {}
    int input_width;
    int input_height;

    int num_thread = 1;

    float net_scale = 1.2;
    float face_threshold = 0.75;
    int min_face_size = 20;
};

class Face3d : public TNNSDKSample {
public:
    virtual ~Face3d() {};
    
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    // virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat,
    //                                                         std::string name = kTNNSDKDefaultName);
    virtual Status Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output);

    bool SetFaceRegion(float x1, float y1, float x2, float y2) {
        bool isValidFace = IsValidFace(x1, y1, x2, y2);
        if(!isValidFace)
            return false;
        
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;

        return true;
    }

    std::shared_ptr<TNN_NS::Mat> GetPrePts() {
        return this->pre_pts;
    }

    bool GetPrevFace() {
        return this->prev_face;
    }
    void SetPrevFace(bool b) {
        this->prev_face = b;
    }

    void SetPrePts(std::shared_ptr<Mat> p, bool deep_copy = false) {
        if(deep_copy) {
            this->pre_pts = std::make_shared<TNN_NS::Mat>(p->GetDeviceType(), p->GetMatType(), p->GetDims());
            auto count = TNN_NS::DimsVectorUtils::Count(p->GetDims());
            memcpy(this->pre_pts->GetData(), p->GetData(), sizeof(float)*count);
        } else {
            this->pre_pts = p;
        }
    }
    
private:
    //prep-rocessing methods
    std::shared_ptr<TNN_NS::Mat> WarpByRect(std::shared_ptr<TNN_NS::Mat> image, float x1, float y1, float x2, float y2, int net_width, float enlarge, std::vector<float>&M);
    
    std::shared_ptr<TNN_NS::Mat> AlignN(std::shared_ptr<TNN_NS::Mat> image, std::shared_ptr<TNN_NS::Mat> pre_pts, std::vector<float>mean, int net_h, int net_w, float net_scale, std::vector<float>&M);
    
    // methods used in pre-processing and post-processing
    std::shared_ptr<TNN_NS::Mat> BGRToGray(std::shared_ptr<TNN_NS::Mat> bgr_mat);
    
    std::vector<float> MatrixInverse2x3(std::vector<float>& mat, int rows, int cols, bool transMat=true);
    
    void LandMarkWarpAffine(std::shared_ptr<TNN_NS::Mat>pts, std::vector<float>& M);
    
    void MatrixMean(const float *ptr, unsigned int rows, unsigned int cols, int axis, std::vector<float>& means);
    
    void MatrixStd(const float *ptr, unsigned int rows, unsigned int cols,int axis, std::vector<float>& stds);
    
    void MatrixSVD2x2(const std::vector<float>a, int rows, int cols, std::vector<float>&u, std::vector<float>&vt);

    bool IsValidFace(float x1, float y1, float x2, float y2) {
        return (x2 - x1 >= min_face_size) && (y2-y1 >= min_face_size);
    }

private:

    // input shape
    int image_w;
    int image_h;
    // whether faces in the previous frame
    bool prev_face = false;
    // face region
    float x1, y1, x2, y2;
    // the minimum face size
    float min_face_size = 20;
    // the confident threshold
    float face_threshold = 0.5;
    // model configs
    float net_scale;
    std::vector<float> mean;
    // current pts data
    std::shared_ptr<TNN_NS::Mat> pre_pts;
    // warpAffine trans matrix
    std::vector<float> M;

    void GenerateLandmarks(std::vector<Face3dInfo> &detects, Mat &landmarks,
                      int image_w, int image_h);

    int num_keypoints = 68;
};

}

#endif // TNN_EXAMPLES_BASE_BLAZEFACE_DETECTOR_H_
