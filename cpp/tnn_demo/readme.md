# TNN Face3d demo  

## set up

### 1. clone project

```bash
git clone https://github.com/lyyiangang/3ddfav2_cpp.git
```

If so, you need download [TNN](https://github.com/Tencent/TNN) and Recompile.

```bash

git clone https://github.com/Tencent/TNN.git

git checkout feature_demo_stream

cd TNN/examples/linux/x86

bash build_linux_native.sh
```
Put libTNN.so and include to our directory
```bash
cd 3ddfav2_cpp

cp TNN/build/libTNN.so.0.1.0.0 cpp/third_party/TNN/libs/
cp TNN/include cpp/third_party/TNN/ -r

cd cpp/third_party/TNN/libs
ln -s libTNN.so.0.1.0.0 libTNN.so.0.1
ln -s libTNN.so.0.1 libTNN.so
```
### 2. convert tnn

you can "pytorch -> onnx ->  tnn" 

this is "onnx -> tnn"
put face3d.onnx to TNN directory
```bash
cd TNN

python converter.py  onnx2tnn ../../face3d.onnx
```
get the tnnmodel and tnnproto
put 3ddfav2_cpp/models

### 3. compile demo
```bash
cd cpp/tnn_demo
mkdir build
cd build
cmake .. && make -j8
```
run tnn_demo
```bash
./tnn_demo ../../../data/facealigner_test.jpg
```

![](results/tnn_det_result2.png)


