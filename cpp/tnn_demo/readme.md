# TNN facealigner demo  

## set up

### 1. clone project

```bash
git clone https://github.com/lyyiangang/3ddfav2_cpp.git
```

If so, you need download [TNN](https://github.com/Tencent/TNN) and Recompile.

```bash
git checkout feature_demo_stream

cd TNN/examples/linux/x86

bash build_linux_native.sh
```
Put libTNN.so to our directory
```bash
cd 3ddfav2_cpp

cp TNN/build/libTNN.so.0.1.0.0 cpp/third_party/TNN/libs/

ln -s libTNN.so.0.1.0.0 libTNN.so.0.1

ln -s libTNN.so.0.1 libTNN.so
```
### 2. convert tnn

This demo is tnn examples, so you can download model in
   
[blazeface.tnnproto](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface.tnnproto)  
[blazeface.tnnmodel](https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/blazeface/blazeface.tnnmodel)  
[blazeface_anchors](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/blazeface/blazeface_anchors.txt)  

[youtu_face_alignment_phase1.tnnproto](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase1.tnnproto)  
[youtu_face_alignment_phase1.tnnmodel](https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase1.tnnmodel)  
[youtu_face_alignment_phase2.tnnproto](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase2.tnnproto)  
[youtu_face_alignment_phase2.tnnmodel](https://media.githubusercontent.com/media/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_face_alignment_phase2.tnnmodel)  

[youtu_mean_pts_phase1.txt](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_mean_pts_phase1.txt)  
[youtu_mean_pts_phase2.txt](https://raw.githubusercontent.com/darrenyao87/tnn-models/master/model/youtu_face_alignment/youtu_mean_pts_phase2.txt)  

Download model put models/facealigner

### 3. compile demo
```bash
cd cpp/tnn_demo
mkdir build
cd build
cmake .. && make -j8
```
run tnn_demo
```bash
./tnn_demo
```
input 

1.picture

2.dir path  ../../../data/facealigner_test.jpg

predictions.bmp will generated in the current directory

![](results/facealigner_result.bmp)


