import tensorflow as tf
from PIL import Image
import requests
from transformers import CLIPProcessor, TFCLIPModel


model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], 
    images=image, return_tensors="tf", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
probs.numpy()

