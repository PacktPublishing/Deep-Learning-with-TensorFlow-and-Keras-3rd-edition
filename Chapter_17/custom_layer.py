# -*- coding: utf-8 -*-
"""packt-18-custom-layer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bD35UW02Quufx4n-eBN1RZzrM_Fyfx1_

# Custom GNN Layer

Tensorflow and DGL re-implementation of PyTorch and DGL example at [Write your own GNN module](https://docs.dgl.ai/tutorials/blitz/3_message_passing.html#sphx-glr-tutorials-blitz-3-message-passing-py).
"""

# Commented out IPython magic to ensure Python compatibility.
# %env DGLBACKEND=tensorflow

!pip install dgl

import dgl
import dgl.data
import dgl.function as fn
import tensorflow as tf

"""## Message passing and GNNs"""

class CustomGraphSAGE(tf.keras.layers.Layer):
  """Graph convolution module used by the GraphSAGE model.

  Parameters
  ----------
  in_feat : int
      Input feature size.
  out_feat : int
      Output feature size.
  """
  def __init__(self, in_feat, out_feat):
    super(CustomGraphSAGE, self).__init__()
    # A linear submodule for projecting the input and neighbor feature to the output.
    self.linear = tf.keras.layers.Dense(out_feat, activation=tf.nn.relu)

  def call(self, g, h):
      """Forward computation

      Parameters
      ----------
      g : Graph
          The input graph.
      h : Tensor
          The input node feature.
      """
      with g.local_scope():
        g.ndata["h"] = h
        # update_all is a message passing API.
        g.update_all(message_func=fn.copy_u('h', 'm'),
                     reduce_func=fn.mean('m', 'h_N'))
        h_N = g.ndata['h_N']
        h_total = tf.concat([h, h_N], axis=1)
        return self.linear(h_total)

class CustomGNN(tf.keras.Model):
  def __init__(self, g, in_feats, h_feats, num_classes):
    super(CustomGNN, self).__init__()
    self.g = g
    self.conv1 = CustomGraphSAGE(in_feats, h_feats)
    self.relu1 = tf.keras.layers.Activation(tf.nn.relu)
    self.conv2 = CustomGraphSAGE(h_feats, num_classes)

  def call(self, in_feat):
    h = self.conv1(self.g, in_feat)
    h = self.relu1(h)
    h = self.conv2(self.g, h)
    return h

"""## Training Loop"""

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
g

LEARNING_RATE = 1e-2
NUM_EPOCHS = 200

def evaluate(model, features, labels, mask, edge_weights=None):
  if edge_weights is None:
    logits = model(features, training=False)
  else:
    logits = model(features, edge_weights, training=False)
  logits = logits[mask]
  labels = labels[mask]
  indices = tf.math.argmax(logits, axis=1)
  acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
  return acc.numpy().item()


def train(g, model, optimizer, loss_fcn, num_epochs, use_edge_weights=False):
  features = g.ndata["feat"]
  labels = g.ndata["label"]
  if use_edge_weights:
    edge_weights = g.edata["w"]

  train_mask = g.ndata["train_mask"]
  val_mask = g.ndata["val_mask"]
  test_mask = g.ndata["test_mask"]
  for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
      if not use_edge_weights:
        logits = model(features)
      else:
        logits = model(features, edge_weights)

      loss_value = loss_fcn(labels[train_mask], logits[train_mask])
      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

    if not use_edge_weights:
      acc = evaluate(model, features, labels, val_mask)
    else:
      acc = evaluate(model, features, labels, val_mask, edge_weights=edge_weights)
    if epoch % 10 == 0:
      print("Epoch {:5d} | loss: {:.3f} | val_acc: {:.3f}".format(
          epoch, loss_value.numpy().item(), acc))

  if not use_edge_weights:
    acc = evaluate(model, features, labels, test_mask)
  else:
    acc = evaluate(model, features, labels, test_mask, edge_weights=edge_weights)
  print("Test accuracy: {:.3f}".format(acc))

model = CustomGNN(g, g.ndata['feat'].shape[1], 16, dataset.num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train(g, model, optimizer, loss_fcn, NUM_EPOCHS)

"""## More customization"""

class CustomWeightedGraphSAGE(tf.keras.layers.Layer):
  """Graph convolution module used by the GraphSAGE model with edge weights.

  Parameters
  ----------
  in_feat : int
      Input feature size.
  out_feat : int
      Output feature size.
  """
  def __init__(self, in_feat, out_feat):
    super(CustomWeightedGraphSAGE, self).__init__()
    # A linear submodule for projecting the input and neighbor feature to the output.
    self.linear = tf.keras.layers.Dense(out_feat, activation=tf.nn.relu)

  def call(self, g, h, w):
    """Forward computation

    Parameters
    ----------
    g : Graph
        The input graph.
    h : Tensor
        The input node feature.
    w : Tensor
        The edge weight.
    """
    with g.local_scope():
      g.ndata['h'] = h
      g.edata['w'] = w
      g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'),
                   reduce_func=fn.mean('m', 'h_N'))
      h_N = g.ndata['h_N']
      h_total = tf.concat([h, h_N], axis=1)
      return self.linear(h_total)

class CustomWeightedGNN(tf.keras.Model):
  def __init__(self, g, in_feats, h_feats, num_classes):
    super(CustomWeightedGNN, self).__init__()
    self.g = g
    self.conv1 = CustomWeightedGraphSAGE(in_feats, h_feats)
    self.relu1 = tf.keras.layers.Activation(tf.nn.relu)
    self.conv2 = CustomWeightedGraphSAGE(h_feats, num_classes)

  def call(self, in_feat, edge_weights):
    h = self.conv1(self.g, in_feat, edge_weights)
    h = self.relu1(h)
    h = self.conv2(self.g, h, edge_weights)
    return h

g.edata["w"] = tf.cast(
    tf.random.uniform((g.num_edges(), 1), minval=3, maxval=10, dtype=tf.int32),
    dtype=tf.float32)
g.edata["w"]

model = CustomWeightedGNN(g, g.ndata['feat'].shape[1], 16, dataset.num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train(g, model, optimizer, loss_fcn, NUM_EPOCHS, use_edge_weights=True)

"""## Even more customization by User Defined Function

Not tensorflow related, but useful to mention.
"""

def u_mul_e_udf(edges):
  return {'m' : edges.src['h'] * edges.data['w']}

def mean_udf(nodes):
  return {'h_N': nodes.mailbox['m'].mean(1)}

