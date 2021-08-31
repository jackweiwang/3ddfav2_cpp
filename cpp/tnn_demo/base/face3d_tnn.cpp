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

// std::shared_ptr<Mat> Face3d::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
//                                                                    std::string name) {

//     return TNNSDKSample::ResizeToInputShape(input_mat, name);
// }

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
    
    
    auto point = output->GetMat("output");
    // TNN_NS::Mat landmarks = *(point.get());
    // float *landmark_data = static_cast<float*>(landmarks.GetData());
    // std::cout << landmark_data[0] << std::endl;

    RETURN_VALUE_ON_NEQ(!point, false,
                           Status(TNNERR_PARAM_ERR, "point mat is nil"));
 
    std::vector<Face3dInfo> bbox_collection;
    GenerateLandmarks(bbox_collection, *(point.get()), option->input_width, option->input_height);
    std::cout << bbox_collection.size() << std::endl;
    output->face_list = bbox_collection;
    
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
        std::shared_ptr<Mat> input1 = nullptr;

        // use face region from face detector
        input1 = WarpByRect(input_mat, x1, y1, x2, y2, image_h, net_scale, M);
std::cout << "input1"<< input1->GetHeight() << input1->GetWidth() <<  std::endl;
        std::cout << "22222222" << std::endl;

        // Normalize
        auto input_convert_param = GetConvertParamForInput();
        status = instance_->SetInputMat(input1, input_convert_param);
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

/*
  wrap the face detected by the face detector rect to input
 */
std::shared_ptr<TNN_NS::Mat> Face3d::WarpByRect(std::shared_ptr<TNN_NS::Mat> image, float x1, float y1, float x2, float y2, int net_width, float enlarge, std::vector<float>&M) {
    float xmin = x1;
    float xmax = x2;
    float ymin = y1;
    float ymax = y2;
    // drop forehead
    ymin  = ymin + (ymax - ymin) * 0.3;
    
    float width = (std::max)(xmax - xmin, ymax - ymin) * enlarge;
    if(width == 0)
        width = 2.0;
    
    float scale = static_cast<float>(net_width) / width;
    
    float cx = (xmax + xmin) / 2.0;
    float cy = (ymax + ymin) / 2.0;
    
    float xOffset = -(cx - width / 2.0);
    float yOffset = -(cy - width / 2.0);
    
    // prepare the warpAffne transformation matrix
    M.resize(2*3, 0);
    float* transM = static_cast<float*>(&M[0]);
    transM[0 * 3 + 0] = scale;
    transM[0 * 3 + 1] = 0.0f;
    transM[0 * 3 + 2] = xOffset * scale;
    transM[1 * 3 + 0] = 0.0f;
    transM[1 * 3 + 1] = scale;
    transM[1 * 3 + 2] = yOffset * scale;
    
    auto transMatDims = image->GetDims();
    transMatDims[2] = net_width;
    transMatDims[3] = net_width;
    
    auto transMat = std::make_shared<TNN_NS::Mat>(image->GetDeviceType(), image->GetMatType(), transMatDims);
    
    TNN_NS::Status status = TNN_OK;
    
    //perform warpAffine
    void* command_queue = nullptr;
    status = instance_->GetCommandQueue(&command_queue);
    if(status != TNN_OK) {
        LOGE("GetCommandQueue Error:%s\n", status.description().c_str());
        return nullptr;
    }
    WarpAffineParam param;
    param.border_type = BORDER_TYPE_CONSTANT;
    param.interp_type = INTERP_TYPE_LINEAR;
    param.border_val = 0;
    memcpy(param.transform, transM, sizeof(float)*M.size());

    status = MatUtils::WarpAffine(*(image.get()), *(transMat.get()), param, command_queue);
    if (status != TNN_OK) {
        LOGE("WarpAffine Error:%s\n", status.description().c_str());
        return nullptr;
    }
    
    return transMat;
}
/*
 warp the input for the phase2 model and the input for the phase1 model if prev_face exists
 parameters:
    @image:     the original image from video, 1280*720*3
    @pre_pts:   landmarks from previous prediction, 117*2 for phase1, 76*2 for phase2
    @mean:      mean landmarks defined by models. 76*2， constant
    @net_h:     the input height for model, constant
    @net_w:     the input width for model, constant
    @net_scale: scale for the input of the model, constant
    @M:         the warpAffine transformation matrix, set by this method
returns:
    the image sent to the phase2 model
 */
std::shared_ptr<TNN_NS::Mat> Face3d::AlignN(std::shared_ptr<TNN_NS::Mat> image, std::shared_ptr<TNN_NS::Mat> pre_pts, std::vector<float> mean_pts, int net_h, int net_w, float net_scale, std::vector<float>&M) {
    DimsVector dims(3, 0);
    // check shape
    const int batch = pre_pts->GetDim(0);
    dims[0] = batch;

    auto channel = pre_pts->GetDim(1);
    dims[1] = channel / 2;
    dims[2] = 2;

    if (TNN_NS::DimsVectorUtils::Count(dims) != mean_pts.size()){
        // shapes not matching, return
        return nullptr;
    }
    
    float *pre_pts_data = static_cast<float*>(pre_pts->GetData());
    float *mean_pts_data = &(mean_pts[0]);
    
    float dx = (net_scale * net_w - net_w) / 2.0;
    float dy = (net_scale * net_h - net_h) / 2.0;
    
    for(int i=0; i<mean_pts.size(); i+=2) {
        mean_pts[i]   = (mean_pts[i]   + dx) / net_scale;
        mean_pts[i+1] = (mean_pts[i+1] + dy) / net_scale;
    }
    
    std::vector<float> pre_pts_mean;
    std::vector<float> mean_pts_mean;
    
    auto rows = dims[1];
    auto cols = dims[2];
    
    // compute mean
    MatrixMean(pre_pts_data, rows, cols, 0, pre_pts_mean);
    MatrixMean(mean_pts_data, rows, cols, 0, mean_pts_mean);
    
    // sub mean
    for(int r=0; r<rows; ++r) {
        for(int c=0; c<cols; ++c) {
            pre_pts_data[r * cols + c]  -= pre_pts_mean[c];
            mean_pts_data[r * cols + c] -= mean_pts_mean[c];
        }
    }
    // compute std
    std::vector<float> pre_pts_std;
    std::vector<float> mean_pts_std;
    MatrixStd(pre_pts_data, rows, cols, -1, pre_pts_std);
    MatrixStd(mean_pts_data, rows, cols, -1, mean_pts_std);
    
    // normalize
    for(int r=0; r<rows; ++r) {
        for(int c=0; c<cols; ++c) {
            pre_pts_data[r * cols + c]  /= pre_pts_std[0];
            mean_pts_data[r * cols + c] /= mean_pts_std[0];
        }
    }
    // svd
    // 1) matmul(pre_pts.T, mean_pts)
    std::vector<float> mul(cols*cols, 0);
    for(int c1 =0; c1<cols; ++c1){
        for(int c2 = 0; c2<cols; ++c2){
            for(int r=0; r<rows; ++r){
                mul[c1 * cols + c2] += pre_pts_data[r * cols + c1] * mean_pts_data[r * cols + c2];
            }
        }
    }
    // 2) get svd result
    std::vector<float> u_mul;
    std::vector<float> vt_mul;
    MatrixSVD2x2(mul, cols, cols, u_mul, vt_mul);
    // 3) reconstruct r by (u_mul*vt_ul).T
    std::vector<float> r_mat(mul.size(), 0);
    for(int c1=0; c1<cols; ++c1){
        for(int c2=0; c2<cols; ++c2){
            for(int k=0; k<cols; ++k){
                r_mat[c1 * cols + c2] += u_mul[c2 * cols + k] * vt_mul[k * cols + c1];
            }
        }
    }
    // compute the warpAffine transformation matrix
    constexpr unsigned int trans_matrix_rows = 2;
    constexpr unsigned int trans_matrix_cols = 3;
    M.resize(6, 0);
    for(int r=0; r<trans_matrix_rows; ++r){
        for(int c=0; c<trans_matrix_cols; ++c){
            float val = 0;
            if(c < trans_matrix_cols - 1){
                val = (mean_pts_std[0] / pre_pts_std[0]) * r_mat[r * cols + c];
                
            } else{
                float val0 = M[r * trans_matrix_cols + 0];
                float val1 = M[r * trans_matrix_cols + 1];
                val = mean_pts_mean[r] - (val0 * pre_pts_mean[0] + val1 * pre_pts_mean[1]);
            }
            M[r * trans_matrix_cols + c] = val;
        }
    }
    //perform warpAffine
    auto transMatDims = image->GetDims();
    transMatDims[2] = net_h;
    transMatDims[3] = net_w;
    auto transMat = std::make_shared<TNN_NS::Mat>(image->GetDeviceType(), image->GetMatType(), transMatDims);
    
    Status status = TNN_OK;
    void* command_queue = nullptr;
    status = instance_->GetCommandQueue(&command_queue);
    if(status != TNN_OK) {
        LOGE("GetCommandQueue Error:%s\n", status.description().c_str());
        return nullptr;
    }
    WarpAffineParam param;
    param.border_type = BORDER_TYPE_CONSTANT;
    param.interp_type = INTERP_TYPE_LINEAR;
    param.border_val = 0;
    float* transM = &(M[0]);
    memcpy(param.transform, transM, sizeof(float)*M.size());

    status = MatUtils::WarpAffine(*(image.get()), *(transMat.get()), param, command_queue);
    if(status != TNN_OK) {
        LOGE("WarpAffine Error:%s\n", status.description().c_str());
        return nullptr;
    }
    
    return transMat;
}

// change BGR Mat to Gray Mat
std::shared_ptr<TNN_NS::Mat> Face3d::BGRToGray(std::shared_ptr<TNN_NS::Mat> bgr_image) {
    Status status = TNN_OK;

    ColorConversionType cvt_type;

    if(bgr_image->GetMatType() == N8UC4) {
        cvt_type = COLOR_CONVERT_BGRATOGRAY;
    } else if (bgr_image->GetMatType() == N8UC3) {
        cvt_type = COLOR_CONVERT_BGRTOGRAY;
    } else {
        return nullptr;
    }

    // only arm supports bgr2gray for now, construct arm input mat when necessary
    TNN_NS::DeviceType src_device_type = bgr_image->GetDeviceType();
    TNN_NS::DeviceType dst_device_type = bgr_image->GetDeviceType();

    std::shared_ptr<TNN_NS::Mat> bgrInputMat = nullptr;
    if (DEVICE_ARM == src_device_type || DEVICE_NAIVE == src_device_type) {
        bgrInputMat = bgr_image;
    } else if (DEVICE_METAL == src_device_type) {
        // condtruct an arm mat
        dst_device_type = DEVICE_ARM;
        bgrInputMat = std::make_shared<TNN_NS::Mat>(dst_device_type, bgr_image->GetMatType(), bgr_image->GetDims());
        status = Copy(bgr_image, bgrInputMat);
        if (status != TNN_OK) {
            LOGE("Copy bgrInput Error:%s\n", status.description().c_str());
            return nullptr;
        }
    }

    auto grayDims = bgrInputMat->GetDims();
    grayDims[1] = 1;
    auto grayMat = std::make_shared<TNN_NS::Mat>(dst_device_type, TNN_NS::NGRAY, grayDims);
    
    void* command_queue = nullptr;
    status = instance_->GetCommandQueue(&command_queue);
    if(status != TNN_OK) {
        LOGE("GetCommandQueue Error:%s\n", status.description().c_str());
        return nullptr;
    }
    status = MatUtils::CvtColor(*(bgrInputMat.get()), *(grayMat.get()), cvt_type, command_queue);
    if(status != TNN_OK) {
        LOGE("CvtColor error:%s\n", status.description().c_str());
        return nullptr;
    }

    // copy when necessary
    std::shared_ptr<TNN_NS::Mat> outputMat = nullptr;
    if (DEVICE_ARM == src_device_type || DEVICE_NAIVE == src_device_type ) {
        outputMat = grayMat;
    } else if (DEVICE_METAL == src_device_type ) {
        // convert ngray mat to nchw_float
        auto grayMatFloat = std::make_shared<TNN_NS::Mat>(dst_device_type, TNN_NS::NCHW_FLOAT, grayMat->GetDims());
        float* grayFloatData  = static_cast<float*>(grayMatFloat->GetData());
        uint8_t* grayUintData = static_cast<uint8_t*>(grayMat->GetData());
        for(int i=0; i<grayDims[2]*grayDims[3]; ++i) {
            grayFloatData[i] = grayUintData[i];
        }
        // copy cpu grat mat to metal
        outputMat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::NCHW_FLOAT, grayMatFloat->GetDims());
        status = Copy(grayMatFloat, outputMat);
        if (status != TNN_OK) {
            LOGE("Copy grayOutput Error:%s\n", status.description().c_str());
            return nullptr;
        }
    }

    return outputMat;
}

/*
 Compute the inverse matrix for 2*3 warpAffine trans matrix
*/
std::vector<float> Face3d::MatrixInverse2x3(std::vector<float>& m, int rows, int cols, bool transMat) {
    std::vector<float> inv(rows*cols, 0);
    if (!transMat) {
        return inv;
    }
    if (rows !=2 || cols != 3) {
        return inv;
    }
    float d   = m[0] * m[4] - m[1] * m[3];
    d          = d != 0 ? 1. / d : 0;

    float a11 = m[4] * d, a22 = m[0] * d;
    inv[0]      = a11;
    inv[1]      = m[1] * (-d);
    inv[3]      = m[3] * (-d);
    inv[4]      = a22;

    float b1 = -a11 * m[2] - inv[1] * m[5];
    float b2 = -inv[3] * m[2] - a22 * m[5];
    inv[2]      = b1;
    inv[5]      = b2;
    
    return inv;
}

/*
 perform warpAffine on pts
 the maximum diff of pts before transform <=: phase1: /phase2: 0.41
 the maximum diff of pts after transform <=: phase1: /phase2: 0.78
 */
void Face3d::LandMarkWarpAffine(std::shared_ptr<TNN_NS::Mat> pts, std::vector<float> &M) {
    constexpr int N = 3;
    
    auto dims = pts->GetDims();
    int pts_dim = 2;
    int num_pts = dims[1] / pts_dim;
    float* pts_data = static_cast<float*>(pts->GetData());
    
    for(int n=0; n<num_pts; ++n){
        float x = M[0 * N + 0] * pts_data[n * pts_dim + 0] + M[0 * N + 1] * pts_data[n * pts_dim + 1] + M[0 * N + 2];
        float y = M[1 * N + 0] * pts_data[n * pts_dim + 0] + M[1 * N + 1] * pts_data[n * pts_dim + 1] + M[1 * N + 2];
        
        pts_data[n * pts_dim + 0] = x;
        pts_data[n * pts_dim + 1] = y;
    }
}

/*
 Compute the means of matrix along the axis
 Parameters:
    @ptr: the pointer to the matrix, data should be stored in ptr following a row-major layout contigunously
    @rows, cols: the shape of matrix
    @axis: which axis to compute mean, '0' for rows, '1' for cols, -1 for all
    @means: vector to store the results
 */
void Face3d::MatrixMean(const float *ptr, unsigned int rows, unsigned int cols, int axis, std::vector<float>& means) {
    unsigned int step_size = 0;
    unsigned int steps = 0;
    unsigned int stride = 0;
   
    means.clear();
    if(axis == 0){
        means.resize(cols);
        step_size = cols;
        steps = rows;
        stride = 1;
    } else if(axis == 1){
        means.resize(rows);
        step_size = 1;
        steps = cols;
        stride = cols;
    } else if(axis == -1){
        means.resize(1);
        step_size = 1;
        steps = rows*cols;
        stride = 1;
    }else{
        return;
    }
    // sum
    for(int s=0; s<steps; ++s){
        for(int n=0; n<means.size(); ++n){
            means[n] += ptr[n * stride + s * step_size];
        }
    }
    // mean
    for(int i=0; i<means.size(); ++i){
        means[i] /= steps;
    }
}


void Face3d::MatrixStd(const float *ptr, unsigned int rows, unsigned int cols, int axis, std::vector<float>& stds) {
    stds.clear();
    if(axis != -1){
        // not supported, return
        return;
    }
    
    stds.resize(1);
    std::vector<float> mean;
    MatrixMean(ptr, rows, cols, -1, mean);
    
    double sum = 0;
    auto count = rows*cols;
    for(int i=0; i<count; ++i) {
        sum += std::pow(std::abs(ptr[i] - mean[0]), 2);
    }
    double std = std::sqrt(sum / count);
    stds[0] = static_cast<float>(std);
}

// svd for 2-by-2 matrix
void Face3d::MatrixSVD2x2(const std::vector<float>a, int rows, int cols, std::vector<float>&u, std::vector<float>&vt) {
    u.clear();
    vt.clear();
    if (rows != 2 || cols != 2) {
        // not supported
        return;
    }
    u.resize(2*2, 0);
    vt.resize(2*2, 0);

    float s[2];
    float v[4];

    s[0] = (sqrt(pow(a[0] - a[3], 2) + pow(a[1] + a[2], 2)) + sqrt(pow(a[0] + a[3], 2) + pow(a[1] - a[2], 2))) / 2;
    s[1] = fabs(s[0] - sqrt(pow(a[0] - a[3], 2) + pow(a[1] + a[2], 2)));
    v[2] = (s[0] > s[1]) ? sin((atan2(2 * (a[0] * a[1] + a[2] * a[3]), a[0] * a[0] - a[1] * a[1] + a[2] * a[2] - a[3] * a[3])) / 2) : 0;
    v[0] = sqrt(1 - v[2] * v[2]);
    v[1] = -v[2];
    v[3] = v[0];

    u[0] = (s[0] != 0) ? (a[0] * v[0] + a[1] * v[2]) / s[0] : 1;
    u[2] = (s[0] != 0) ? (a[2] * v[0] + a[3] * v[2]) / s[0] : 0;
    u[1] = (s[1] != 0) ? (a[0] * v[1] + a[1] * v[3]) / s[1] : -u[2];
    u[3] = (s[1] != 0) ? (a[2] * v[1] + a[3] * v[3]) / s[1] : u[0];
    // transpose
    vt[0] = v[0];
    vt[1] = v[2];
    vt[2] = v[1];
    vt[3] = v[3];
}

}
