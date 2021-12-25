import tensorflow as tf
import tensorflow_hub as hub

embed = hub.KerasLayer(
    "https://tfhub.dev/google/elmo/3",
    input_shape=[],     # Expects a tensor of shape [batch_size] as input.
    dtype=tf.string)    # Expects a tf.string input tensor.
model = tf.keras.Sequential([embed])
model.summary()

embeddings = model.predict([
   "i i like green eggs and ham",
   "would you eat them in a box"
])
print(embeddings.shape)
