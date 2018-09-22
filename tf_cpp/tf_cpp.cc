#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <vector>
#include <iostream>
#include <chrono>

#define WARMUP_BATCH_SIZE 100

using namespace tensorflow;

Status LoadGraph(string graph_file_name, Session** session)
{
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(
        ReadBinaryProto(Env::Default(), graph_file_name, &graph_def));
    TF_RETURN_IF_ERROR(NewSession(SessionOptions(), session));
    TF_RETURN_IF_ERROR((*session)->Create(graph_def));

    return Status::OK();
};

std::vector<std::pair<string, tensorflow::Tensor>> MakeInputs(int batch_size)
{
    Tensor input(DT_FLOAT, TensorShape({batch_size, 25}));

    auto data_ = input.flat<float>().data();
    for (int i = 0; i < batch_size * 25; ++i)
      data_[i] = 0.42;

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        { "Input_0", input },
    };

    return inputs;
};

int main(int argc, char const *argv[])
{
    Session* session;
    TF_CHECK_OK(
        LoadGraph(argv[1], &session));

    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session->Run(MakeInputs(WARMUP_BATCH_SIZE), {"Output_0/Softmax"}, {}, &outputs));

    for (int batch_size : {1, 8, 32, 256, 1024}) {
        auto inputs = MakeInputs(batch_size);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i) {
            TF_CHECK_OK(session->Run(inputs, {"Output_0/Softmax"}, {}, &outputs));
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "TF_CPP," << argv[1] << ","
        << batch_size << ","
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 100. << ","
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / (100. * batch_size)
        << std::endl;

    }

    session->Close();
    return 0;
}
