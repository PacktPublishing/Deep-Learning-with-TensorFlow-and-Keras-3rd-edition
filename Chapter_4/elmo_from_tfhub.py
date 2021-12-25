import tensorflow as tf
import tensorflow_hub as hub

elmo = hub.load("https://tfhub.dev/google/elmo/3")
embeddings = elmo.signatures["default"](
    tf.constant([
      "i like green eggs and ham",
      "would you eat them in a box"
    ]))["elmo"]
print(embeddings.shape)

