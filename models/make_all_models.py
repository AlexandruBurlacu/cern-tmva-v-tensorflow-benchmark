import tensorflow as tf

max_num_of_layers = 5
all_neurons = [16, 32, 128, 256, 1024, 4096, 15, 30, 130, 250, 1000, 4000]

for layers in range(max_num_of_layers):
  for neurons in all_neurons:
    fcn = tf.keras.layers.Input(shape=(25,), name="Input_0")
    inp = fcn
    fcn = tf.keras.layers.Dense(neurons, activation="relu")(fcn)
    for _ in range(layers):
      fcn = tf.keras.layers.Dense(neurons, activation="relu")(fcn)
    final = tf.keras.layers.Dense(10, activation="softmax", name="Output_0")(fcn)

    model = tf.keras.models.Model(inputs=inp, outputs=final)
    model.compile("sgd", "categorical_crossentropy")

    model.save(f"h5/test_model_relu_l{layers+1}_n{neurons}.h5")
    model.summary()
    fcn, inp, model, final = None, None, None, None

