# TMVA vs TensorFlow Benchmark

During the summer of 2018 (2 July - 24 August) I was working on benchmarking TMVA (a toolkit for machine learning, part of CERN's ROOT data analysis framework) versus TensorFlow on inference performance of Multi-Layered Perceptrons on CPUs for High-Energy Physics problems. This project was aimed to clear the roadmap of the TMVA project and understand better how to position it in the HEP community.

You can check the full report [here](https://cds.cern.ch/record/2641377)
Also, there are slides, [here](https://slides.com/alexandruburlacu/benchmarking-tmva-package-against-tensorflow-on-event-by-event-inference-performance-on-multi-layered-perceptrons-for-hep)

### Getting ROOT

Check their installation/building instructions on GitHub, [here](https://github.com/root-project/root#building), the process is actually very easy, just check for the presence of all dependencies. For the benchmark was used ROOT v6.14.00.

### Docker Image
Currently, there's no `Dockerfile` to containerize the environment in which the benchmark was run. But, you can follow the instructions from [this](https://www.pugetsystems.com/labs/hpc/Build-TensorFlow-CPU-with-MKL-and-Anaconda-Python-3-6-using-a-Docker-Container-1133/) blog post to make yourself one. Eventually, maybe, I will add a `Dockerfile` in this repository.

### Making Keras & TF models

1. Run the `python make_all_models.py` from models dir.
2. Run `ls h5/ | xargs -I @ python to_graph_def.py --model-file h5/@` from models dir.

### Running TF and Keras benchmarks, without optimization

Both `keras_benchmark_loop.py` and `tf_benchmark_loop.py` have a common CLI API `python <>_benchmark_loop.py --path <path to the directory with stored models, proto for tf, h5 for Keras>`

Both scripts output CSV rows to stdout. To collect them, add `>> benchmark_results.csv` to the command above.

### Building the optimized TensorFlow

To build TensorFlow, you will need to git clone it and to have bazel installed (use `apt-get` on Ubuntu to install it).

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
# from within TensorFlow repo
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

Again, the CSV rows are outputted to stdout, so to collect them, add `>> benchmark_results.csv`.

