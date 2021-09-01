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

#include "face_det_3d.h"
#include "tnn/utils/dims_vector_utils.h"
#include "blazeface_detector.h"
#include "tnn_sdk_sample.h"
#include "face3d_tnn.h"

#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
namespace TNN_NS {
    
Status FaceDetect3D::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    if (sdks.size() < 2) {
        return Status(TNNERR_INST_ERR, "FaceDetectAligner::Init has invalid sdks, its size < 2");
    }

    predictor_detect_ = sdks[0];
    predictor_3d_ = sdks[1];
    return TNNSDKComposeSample::Init(sdks);
}

Status FaceDetect3D::Predict(std::shared_ptr<TNNSDKInput> sdk_input,
                                  std::shared_ptr<TNNSDKOutput> &sdk_output) {
    Status status = TNN_OK;

    if (!sdk_input || sdk_input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    auto predictor_detect_async = predictor_detect_;
    auto predictor_3d_async = predictor_3d_;
    auto predictor_detect_cast = dynamic_cast<BlazeFaceDetector *>(predictor_detect_async.get());
    auto predictor_mesh_cast = dynamic_cast<Face3d *>(predictor_3d_async.get());

    auto image_mat = sdk_input->GetMat();
    const int image_orig_height = image_mat->GetHeight();
    const int image_orig_width = image_mat->GetWidth();

    // output of each model
    std::shared_ptr<TNNSDKOutput> sdk_output_face = nullptr;
    std::shared_ptr<TNNSDKOutput> sdk_output_mesh = nullptr;

    std::vector<BlazeFaceInfo> face_list;
    std::shared_ptr<TNN_NS::Mat> phase1_pts = nullptr;

            int crop_height = -1, crop_width = -1;
        int crop_x = -1, crop_y = -1;
        std::shared_ptr<TNN_NS::Mat> croped_mat;
        std::shared_ptr<TNN_NS::Mat> resized_mat;
    // phase1: face detector
    {
                // 1) prepare input for phase1 model
        if(!has_prev_face_) {
            status = predictor_detect_cast->Predict(std::make_shared<BlazeFaceDetectorInput>(image_mat), sdk_output_face);
            RETURN_ON_NEQ(status, TNN_OK);

            if (sdk_output_face && dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output_face.get())) {
                auto face_output = dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output_face.get());
                face_list = face_output->face_list;
            }
            std::cout << "000000000" << std::endl;
            if(face_list.size() <= 0) {
                //no faces, return
                printf("Error no faces found!\n");
                return status;
            }
            


            auto face_orig = face_list[0].AdjustToViewSize(image_orig_height, image_orig_width, 2);
            std::cout << face_orig.x1 << face_orig.y1 << face_orig.x2 << face_orig.y2 << std::endl;
            // set face region for phase1 model
            
            if (!(predictor_mesh_cast &&
                    predictor_mesh_cast->SetFaceRegion(face_orig.x1, face_orig.y1, face_orig.x2, face_orig.y2))) {
                //no invalid faces, return
                printf("Error no valid faces found!\n");
                return status;
            }
    }
std::cout << "0" << std::endl;
        // 2) predict
        status = predictor_mesh_cast->Predict(std::make_shared<Face3dInput>(image_mat), sdk_output_mesh);
        RETURN_ON_NEQ(status, TNN_OK);
std::cout << "9999999999" << std::endl;
        // update prev_face
        has_prev_face_ = predictor_mesh_cast->GetPrevFace();
        if(!has_prev_face_) {
            LOGD("Next frame will use face detector!\n");
        }
        phase1_pts = predictor_mesh_cast->GetPrePts();
        // //1.5*crop
        // crop_height =  1.0 * (face_orig.y2 - face_orig.y1);
        // crop_width  =  1.0 * (face_orig.x2 - face_orig.x1);
        
        // crop_x = (std::max)(0.0, face_orig.x1 + 0.05 * crop_width);
        // crop_y = (std::max)(0.0, face_orig.y1 + 0.05 * crop_height);
        // std::cout << "cropx"<< crop_x << crop_y <<  std::endl;

        // crop_width  = (std::min)(crop_width,  image_orig_width - 2 * crop_x);
        // crop_height = (std::min)(crop_height, image_orig_height- 2 * crop_y);


        // std::cout << "crop"<< crop_height << crop_width <<  std::endl;
        // std::cout << "image"<< image_mat->GetHeight() << image_mat->GetWidth() <<  std::endl;
        // DimsVector crop_dims = {1, 3, static_cast<int>(crop_height), static_cast<int>(crop_width)};
        // croped_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC3, crop_dims);
        // status = predictor_mesh_cast->Crop(image_mat, croped_mat, crop_x, crop_y);
        // RETURN_ON_NEQ(status, TNN_OK);

        // DimsVector resize_dims = {1, 3, 120, 120};
        // resized_mat = std::make_shared<TNN_NS::Mat>(croped_mat->GetDeviceType(), TNN_NS::N8UC3, resize_dims);
        // status = predictor_mesh_cast->Resize(croped_mat, resized_mat, TNNInterpLinear);
        // RETURN_ON_NEQ(status, TNN_OK);


        // status = predictor_mesh_cast->Predict(std::make_shared<Face3dInput>(resized_mat), sdk_output_mesh);
        // RETURN_ON_NEQ(status, TNN_OK);


        
    }
    //get output
    
    {
        std::cout << "777777" << std::endl;
        sdk_output = std::make_shared<Face3dOutput>();
        auto phase1_output = dynamic_cast<Face3dOutput *>(sdk_output_mesh.get());

        auto& points_phase = phase1_output->face.key_points_3d;

        auto output = dynamic_cast<Face3dOutput *>(sdk_output.get());
        output->face.key_points_3d = points_phase;
        output->face.image_height = image_orig_height;
        output->face.image_width  = image_orig_width;
    }

    return TNN_OK;
}
}
