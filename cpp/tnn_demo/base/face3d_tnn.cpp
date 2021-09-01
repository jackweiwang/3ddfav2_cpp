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
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    image_h = option->input_height;
    image_w = option->input_width;
    face_threshold = option->face_threshold;
    min_face_size = option->min_face_size;
    prev_face = false;

    net_scale = option->net_scale;
    pre_pts = nullptr;

    return TNN_OK;
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
    
    std::shared_ptr<Mat> pts = nullptr;

    pts = output->GetMat("output");

    std::vector<Face3dInfo> bbox_collection;
    GenerateLandmarks(bbox_collection, *(pts.get()), option->input_width, option->input_height);
    output->face = bbox_collection;
    // prepare output
    // Face3dInfo face;
    // std::cout << "555555555" << std::endl;
    // constexpr int pts_dim = 3;
    // auto pts_data = static_cast<float*>(pts->GetData());

    // for (int i=0; i<num_keypoints; ++i) {
    // //for (int i=0; i<pts_cnt; ++i) {
    //     face.key_points_3d.push_back(std::make_tuple(pts_data[i * pts_dim + 0],
    //                                                  pts_data[i * pts_dim + 1],
    //                                                  pts_data[i * pts_dim + 2]));
    //     std::cout << pts_data[i*3+0] << std::endl;
    // }
    // output->face = std::move(face);

    return status;
}
void Face3d::GenerateLandmarks(std::vector<Face3dInfo> &detects, TNN_NS::Mat &landmarks, int image_w, int image_h ) {

    float *landmark_data = static_cast<float*>(landmarks.GetData());


    Face3dInfo info;
    info.image_width = image_w;
    info.image_height = image_h;
    std::cout << image_w <<  image_h<< std::endl;

    //key points 3d
    std::vector<triple<float, float, float>> key_points_3d;
    for (int i=0; i<num_keypoints; ++i) {
        info.key_points_3d.push_back(std::make_tuple(landmark_data[i * 3 + 0],
                                                     landmark_data[i * 3 + 1],
                                                     landmark_data[i * 3 + 2]));
        //std::cout << landmark_data[i*3+0] << std::endl;
    }


    detects.push_back(std::move(info));
    
    
}

Status Face3d::Predict(std::shared_ptr<TNNSDKInput> input, std::shared_ptr<TNNSDKOutput> &output) {
    Status status = TNN_OK;
    
    if (!input || input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    std::cout << "22222222" << std::endl;
#if TNN_SDK_ENABLE_BENCHMARK
    bench_result_.Reset();
    for (int fcount = 0; fcount < bench_option_.forward_count; fcount++) {
        SampleTimer sample_time;
        sample_time.Start();
#endif
        // step 1. set input mat for phase1
        auto input_names = GetInputNames();
        RETURN_VALUE_ON_NEQ(input_names.size(), 1, Status(TNNERR_PARAM_ERR, "TNNInput number is invalid"));
        std::cout << "22222222" << std::endl;
        auto input_mat = input->GetMat();

        // Normalize
        auto input_convert_param = GetConvertParamForInput();
        status = instance_->SetInputMat(input_mat, input_convert_param);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);


        // step 3. get output mat of phase1 model
        output = CreateSDKOutput();
        auto input_device_type = input_mat->GetDeviceType();
        auto output_names = GetOutputNames();
        for (auto name : output_names) {
            auto output_convert_param = GetConvertParamForOutput(name);
            std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;
            status = instance_->GetOutputMat(output_mat, output_convert_param, name,
                                             TNNSDKUtils::GetFallBackDeviceType(input_device_type));
            RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
            output->AddMat(output_mat, name);
        }

#if TNN_SDK_ENABLE_BENCHMARK
        sample_time.Stop();
        double elapsed = sample_time.GetTime();
        bench_result_.AddTime(elapsed);
#endif
        // post-processing
        ProcessSDKOutput(output);
#if TNN_SDK_ENABLE_BENCHMARK
    }
#endif
    
    return status;
}


}
