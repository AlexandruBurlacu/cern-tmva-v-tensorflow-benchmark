import argparse

import tensorflow as tf


def to_graph_def(filename):
  model = tf.keras.models.load_model(filename)
  sess = tf.keras.backend.get_session()
  proto_file_name = filename.split("/")[-1].split(".")[0] + ".pb"
  minimal_graph = tf.graph_util.convert_variables_to_constants(
                        sess, sess.graph_def, ["Output_0/Softmax"])

  print(proto_file_name)
  tf.train.write_graph(minimal_graph, "proto/",
                       proto_file_name, as_text=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-file")
  args = parser.parse_args()
  to_graph_def(args.model_file)

