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


    RETURN_VALUE_ON_NEQ(!score, false,
                           Status(TNNERR_PARAM_ERR, "score mat is nil"));
 
    std::vector<Face3dInfo> bbox_collection;
    GenerateLandmarks(bbox_collection, *(score.get()), option->input_width, option->input_height);

    output->face_list = bbox_collection;
    
    return status;
}
void Face3d::GenerateLandmarks(std::vector<Face3dInfo> &detects, TNN_NS::Mat &landmarks, int image_w, int image_h ) {

    float *landmark_data = static_cast<float*>(landmarks.GetData());


    Face3dInfo info;
    info.image_width = image_w;
    info.image_height = image_h;


    //key points 3d
    std::vector<triple<float, float, float>> key_points_3d;
    for (int i=0; i<num_keypoints; ++i) {
        info.key_points_3d.push_back(std::make_tuple(landmark_data[i * 3 + 0],
                                                     landmark_data[i * 3 + 1],
                                                     landmark_data[i * 3 + 2]));
    }

    // key points
    // for(int i=0; i<num_keypoints*3; ++i) {
    //     // int offset = j * 3 ;
    //     // float xp = (landmark_data[offset + 0]  ) ;
    //     // float yp = (landmark_data[offset + 1]  ) ;
    //     // float zp = (landmark_data[offset + 2]  ) ;
    //     //info.naive_key_points.push_back( std::make_tuple(xp, yp, zp) );
    //     info.key_points.push_back( landmark_data[i] );
    // }
    detects.push_back(std::move(info));
    
    
}

}
