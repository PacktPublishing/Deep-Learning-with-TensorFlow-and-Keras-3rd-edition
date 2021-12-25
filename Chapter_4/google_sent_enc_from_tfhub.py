import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")
embeddings = embed([
   "i like green eggs and ham",
   "would you eat them in a box"
])["outputs"]
print(embeddings.shape)

