#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <iostream>
#include <string>
#include <limits>


#include "face3d_tnn.h"

#include "macro.h"
#include "utils/utils.h"
#include "flags.h"

using namespace std;
using namespace TNN_NS;

int plot_circle(uint8_t* src, int point[2], int width, int height) {
    const int CIRCLE_RADIUS =1;
    for (int y = -CIRCLE_RADIUS; y < (CIRCLE_RADIUS + 1); ++y) {
        for (int x = -CIRCLE_RADIUS; x < (CIRCLE_RADIUS + 1); ++x) {
            const int xx = point[0] + x;
            const int yy = point[1] + y; 
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                int index  = yy * width + xx;
                src[index * 3] = 255;
                src[index * 3 + 1] = 255;
                src[index * 3 + 2] = 255;
            }
        }
    }
    return 0;
}

// convert to image coordinate system
void post_process(std::vector<float>& pts){
    float min_z = std::numeric_limits<float>::max();
    for(int ii = 0; ii < 68; ++ii){
        pts[ii * 3 + 0] -= 1.0;
        pts[ii * 3 + 2] -= 1.0;
        pts[ii * 3 + 1] = 120.0 - pts[ii * 3 + 1];
        if(pts[ii * 3 + 2] < min_z)
            min_z = pts[ii * 3 + 2];
    }
    for(int ii = 0; ii < 68; ++ii){
        pts[ii * 3 + 2] -= min_z;
    }
}
Status initDetectPredictor(std::shared_ptr<Face3d>& predictor, int argc, char** argv) {
    char detect_path_buff[256];
    char *detect_model_path = detect_path_buff;
    if (argc < 3) {
        strncpy(detect_model_path, "../../../models/", 256);
    } else {
        strncpy(detect_model_path, argv[2], 256);
    }

    std::string detect_proto = std::string(detect_model_path) + "face3d.tnnproto";
    std::string detect_model = std::string(detect_model_path) + "face3d.tnnmodel";
    //std::string anchor_path = std::string(detect_model_path) + "blazeface_anchors.txt";

    auto detect_proto_content = fdLoadFile(detect_proto);
    auto detect_model_content = fdLoadFile(detect_model);
    auto detect_option = std::make_shared<Face3dOption>();

    const int targer_height = 120;
    const int targer_width = 120;
    DimsVector target_dims = {1, 3, targer_height, targer_width};
    {
        detect_option->proto_content = detect_proto_content;
        detect_option->model_content = detect_model_content;
        detect_option->library_path = "";
        
        detect_option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        //#ifdef _CUDA_
        //    detect_option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        //#endif

    }

    predictor = std::make_shared<Face3d>();
    auto status = predictor->Init(detect_option);
    return status;
}

int main(int argc, char **argv){
    Status status = TNN_OK;
    std::shared_ptr<Face3d> predictor;
    CHECK_TNN_STATUS(initDetectPredictor(predictor, argc, argv));

    std::string img_name = argv[1];

    int originalWidth;
    int originalHeight;
    int originChannel;
    auto inputImage = stbi_load(img_name.c_str(), &originalWidth, &originalHeight, &originChannel, 3);
    if (nullptr == inputImage) {
        cout<<"can not open "<<img_name<<"\n";
        return 1;
    }
    const auto rgbPtr = reinterpret_cast<uint8_t*>(inputImage);
    std::shared_ptr<TNNSDKOutput> output = nullptr;


    DimsVector target_dims = {1, 3, 120, 120};

    DimsVector nchw = {1, originChannel, originalHeight, originalWidth};
    auto image_mat = std::make_shared<Mat>(DEVICE_NAIVE, N8UC3, nchw, rgbPtr);
    auto resize_mat = std::make_shared<Mat>(DEVICE_NAIVE, N8UC3, target_dims);
    // tnn::Mat *test = image_mat.get();
    // tnn::Mat *test1 = resize_mat.get();
    // std::cout << "height"<< test->GetHeight() << test->GetWidth() <<  std::endl;
    // std::cout << "height1"<< test1->GetHeight() << test1->GetWidth() <<  std::endl;
    //TNNInterpNearest TNNInterpLinear
    status = predictor->Resize(image_mat, resize_mat, TNNInterpLinear);
    RETURN_ON_NEQ(status, TNN_OK);
    
    status = predictor->Predict(std::make_shared<Face3dInput>(resize_mat), output);
    RETURN_ON_NEQ(status, TNN_OK);

    //std::vector< float> lnds(68*3);
    predictor->ProcessSDKOutput(output);


    std::vector<Face3dInfo> face_info;
    if (output && dynamic_cast<Face3dOutput *>(output.get()))
    {
        auto face_output = dynamic_cast<Face3dOutput *>(output.get());
 
        face_info = face_output->face_list;
    }
    
    if(face_info.size() <= 0) {
        //no faces, return
        LOGD("Error no faces found!\n");
        return status;
    }

    int pointNum = 68*3;
    float xscale = (float)originalWidth/120;
    float yscale = (float)originalHeight/120;
    auto lnds = face_info[0].naive_key_points;

    post_process(lnds);
    std::vector<float> src_point(lnds);
    for (int i = 0; i < pointNum; ++i) {
        src_point[i * 3 + 0] = lnds[i * 3 + 0] * xscale;
        src_point[i * 3 + 1] = lnds[i * 3 + 1] * yscale;

    }

    for(int ii = 0; ii < 68; ++ii){
        int pt[2] = {src_point[ii * 3], src_point[ii * 3 + 1]};
        plot_circle(rgbPtr, pt, originalWidth, originalHeight);
        //std::cout << lnds[ii] << std::endl;
    }   

    std::string out_name = "tnn_det_result.png";
    stbi_write_png(out_name.c_str(), originalWidth, originalHeight, 3, inputImage, 3 * originalWidth);
    stbi_image_free(inputImage);
    std::cout<<"output detect result to "<< out_name<<std::endl;

    return 0; 
}
