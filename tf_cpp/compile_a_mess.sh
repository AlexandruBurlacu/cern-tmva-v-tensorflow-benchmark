bazel build -c opt \
  --copt=-mavx --copt=-mavx2 --copt=-mavx512f \
  --copt=-mfma --copt=-msse4.2 --copt=-msse4.1 \
  //tensorflow:libtensorflow_cc.so

export LD_LIBRARY_PATH=/root/tf_docker/tensorflow/bazel-bin/tensorflow

g++ --std=c++11 -o tf_cpp tf_cpp.cc \
 -I ./../../ \
 -I ./../../bazel-tensorflow/external/eigen_archive \
 -I ./../../bazel-tensorflow/external/protobuf_archive/src \
 -I ./../../bazel-genfiles \
 -L ./../../bazel-bin/tensorflow \
 -ltensorflow_cc -ltensorflow_framework 

./tf_cpp

