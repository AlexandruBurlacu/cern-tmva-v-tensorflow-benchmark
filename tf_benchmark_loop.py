import tensorflow as tf

import numpy as np
import time
import os
import argparse

# config = tf.ConfigProto()
# config.intra_op_parallelism_threads = 8
# config.inter_op_parallelism_threads = 2



def benchmark(models_path):
    model_paths = [models_path + "/" + model_name for model_name in os.listdir(models_path)]

    batch_sizes = [1, 8, 32, 256, 1024]

    for model_path in model_paths:

        with tf.Session() as sess:

            graph_def = tf.GraphDef()
            with open(model_path, "rb") as graph_fptr:
                graph_def.ParseFromString(graph_fptr.read())
                tf.import_graph_def(graph_def)

            inp = "import/Input_0:0"
            out = "import/Output_0/Softmax:0"
            # warming up the model
            sess.run(out, feed_dict={inp: np.random.random((100, 25))})
            # benchmark loop
            for batch_size in batch_sizes:
                data = np.random.random((batch_size, 25))
                start_time = time.time()
                # to get an average real-world result
                for _ in range(10):
                    sess.run(out, feed_dict={inp: data})
                end_time = time.time() - start_time
                print(f"TF,{model_path},{batch_size},{end_time/10.},{end_time/(10*batch_size)}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()

    benchmark(args.path)

