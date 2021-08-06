#include "face3d_tnn.h"
#include <cmath>
#include <fstream>
#include <cstring>
#include <time.h>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

namespace TNN_NS {

Status Face3d::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<Face3dOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    std::string line;


    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];

    return status;
}

std::shared_ptr<Mat> Face3d::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
    return TNNSDKSample::ResizeToInputShape(input_mat, name);
}

MatConvertParam Face3d::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5, 0.0};
    input_convert_param.bias  = {-1.0, -1.0, -1.0, 0.0};
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> Face3d::CreateSDKOutput() {
    return std::make_shared<Face3dOutput>();
}

Status Face3d::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<Face3dOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                           Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<Face3dOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
    Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    
    auto score = output->GetMat("output");
    //auto boxes  = output->GetMat("544");
    RETURN_VALUE_ON_NEQ(!score, false,
                           Status(TNNERR_PARAM_ERR, "score mat is nil"));
 
    TNN_NS::Mat scores = *(score.get());
    std::vector<float> bbox_collection;

    float *score_data = static_cast<float*>(scores.GetData());

    for(int i=0; i<68; ++i) {

        //float kp_x = score_data[i * 2 + 0] ;
        //float kp_y = score_data[i * 2 + 1] ;
        bbox_collection.push_back(score_data[i]);
    }

    output->face_list = bbox_collection;
    
    return status;
}
}
// Face3d::Face3d(const std::string& model_path){
//     float means[3] = {127.5, 127.5, 127.5};
//     float std[3] = {1/127.5, 1/127.5, 1/127.5};
//     _preprocess.reset(
//         MNN::CV::ImageProcess::create(MNN::CV::ImageFormat::RGB,
//         MNN::CV::BGR, \
//         means, \
//         3, \
//         std, \
//         3)
//     );
//     _interpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
//     MNN::ScheduleConfig cfg;
//     _session = _interpreter->createSession(cfg);
// }

// Face3d::~Face3d(){
// }

// std::vector<float> Face3d::Predict(const uint8_t* buffer_120x120x3){
//     MNN::Tensor* input_ts = _interpreter->getSessionInput(_session, "input");
//     _preprocess->convert(buffer_120x120x3, 120, 120, 0, input_ts);
//     _interpreter->runSession(_session); 
//     MNN::Tensor* lnd_output_ts = _interpreter->getSessionOutput(_session, "output");
//     MNN::Tensor lnd_host(lnd_output_ts, MNN::Tensor::CAFFE);
//     lnd_output_ts->copyToHostTensor(&lnd_host);
//     std::vector<float> output_landmarks(lnd_host.elementSize(), 0);
//     std::copy(lnd_host.host<float>(), lnd_host.host<float>() + lnd_host.elementSize(), output_landmarks.data());
//     return output_landmarks;
// }
