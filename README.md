# TMVA vs TensorFlow Benchmark

### Making Keras & TF models

1. Run the `python make_all_models.py` from models dir.
2. Run `ls h5/ | xargs -I @ python to_graph_def.py --model-file h5/@` from models dir.

### Running TF and Keras benchmarks, without optimization

Both `keras_benchmark_loop.py` and `tf_benchmark_loop.py` have a common CLI API `python <>_benchmark_loop.py --path <path to the directory with stored models, proto for tf, h5 for Keras>`

Both scripts output CSV rows to stdout. To collect them, add `>> benchmark_results.csv` to the command above.

### Building the optimized TensorFlow

To build tensorflow, you will need to git clone it and to have bazel installed (use `apt-get` on Ubuntu to install it).

```bash
bazel build --config=opt --config=mkl \
  --copt=-mavx --copt=-mavx2 --copt=-mavx512f \ # AVX512f only if your processor supports it (Intel Skylake or newer)
  --copt=-mfma --copt=-msse4.2 --copt=-msse4.1 \
  //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package ../tensorflow_pkg
# And then I don't remember, check the TensorFlow's page on how to install from source.
```

Now you're ready to run TF Python API and Keras with optimized TF.
Also, in order to tune MKL, run `source env.sh`, there are some environment variables, you can check their meaning also in the TensorFlow documentation, regarding high-performance tunning of CPU distributions.

### Building and Running the TF C++ API

1. Build the `libtensorflow` as a shared library
```
# from within tensorflow repo
bazel build -c opt \
  --copt=-mavx --copt=-mavx2 --copt=-mavx512f \
  --copt=-mfma --copt=-msse4.2 --copt=-msse4.1 \
  //tensorflow:libtensorflow_cc.so
# No MKL tho
```
2. `export LD_LIBRARY_PATH=/root/tf_docker/tensorflow/bazel-bin/tensorflow`
3. Compile your project
```
g++ --std=c++11 -o tf_cpp tf_cpp.cc \
 -I ./../../ \
 -I ./../../bazel-tensorflow/external/eigen_archive \
 -I ./../../bazel-tensorflow/external/protobuf_archive/src \
 -I ./../../bazel-genfiles \
 -L ./../../bazel-bin/tensorflow \
 -ltensorflow_cc -ltensorflow_framework
```

4. The `tf_cpp` executable should be ran like this `ls <path to proto files> | xargs -I @ ./tf_cpp <path to proto files>/@`

Again, the CSV rows are outputed to stdout, so to collect them, add `>> benchmark_results.csv`.