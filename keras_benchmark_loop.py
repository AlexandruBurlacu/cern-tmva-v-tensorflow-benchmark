import tensorflow as tf
import numpy as np

import time
import os
import argparse

def benchmark(models_path):
    model_paths = [models_path + "/" + model_name for model_name in os.listdir(models_path)]

    batch_sizes = [1, 8, 32, 256, 1024]

    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)
        # warming up the model
        model.predict(np.random.random((100, 25)))
        # benchmark loop
        for batch_size in batch_sizes:
            data = np.random.random((batch_size, 25))
            start_time = time.time()
            # to get an average real-world result
            for _ in range(10):
                model.predict(data)
            end_time = time.time() - start_time
            print(f"KERAS,{model_path},{batch_size},{end_time/10.},{end_time/(10*batch_size)}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()

    benchmark(args.path)

