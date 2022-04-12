from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = TFBertForMaskedLM.from_pretrained("bert-base-cased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="tf")
logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = tf.where(inputs.input_ids == tokenizer.mask_token_id)[0][1]

# predicted_token_id = tf.math.argmax(logits[0, mask_token_index], axis=-1)
predicted_token_id = tf.math.argmax(logits[:, mask_token_index], axis=-1)
tokenizer.convert_ids_to_tokens(predicted_token_id)[0]

