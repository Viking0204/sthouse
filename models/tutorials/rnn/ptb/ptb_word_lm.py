from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = len(data) // batch_size
    self.input_data, self.targets, self.weight = reader.ptb_producer(data, batch_size, num_steps, name=name)
    


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    tag_size = config.tag_size
   
    with tf.device("/cpu:0"):
      # inputs(batch, steps, hidden_size)
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)
    
    # output(batch*steps, 2*hidden_size)
    output = self._build_rnn_graph(inputs, input_.weight, config, is_training)

    softmax_w = tf.get_variable("softmax_w", shape=[2*size, tag_size],
                dtype=tf.float32)

    softmax_b = tf.get_variable("softmax_b", shape=[tag_size], dtype=tf.float32,
                initializer=tf.zeros_initializer())

    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    # logits(batch,steps,tag_size)
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, tag_size])
    
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
    logits, input_.targets, input_.weight)

    loss = tf.reduce_mean(-log_likelihood)

    # Update the cost & get params
    self._cost = loss
    self._tranM = transition_params  
    self._logits = logits 
    self._tar_test = input_.targets
    self._wei_test = input_.weight 
    
    if not is_training:
      return   
    
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

 
  def _build_rnn_graph(self, inputs, sequence_length, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    cell_fw = tf.contrib.rnn.LSTMCell(config.hidden_size, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(config.hidden_size, state_is_tuple=True)
 
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
    cell_bw, inputs , sequence_length = sequence_length,
    dtype=tf.float32)
 
    #outputs(batch*steps, 2*hidden_size)
    with tf.variable_scope("RNN"):
      outputs = tf.concat([output_fw, output_bw], axis=-1)
    outputs = tf.reshape(outputs, [-1, 2 * config.hidden_size])
  
    return outputs

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost,
           util.with_prefix(self._name, "tranM"): self._tranM,
           util.with_prefix(self._name, "logits"): self._logits,
           util.with_prefix(self._name, "tar_test"): self._tar_test,
           util.with_prefix(self._name, "wei_test"): self._wei_test}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    
    for name, op in ops.items():
      tf.add_to_collection(name, op)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    self._tranM = tf.get_collection_ref(util.with_prefix(self._name, "tranM"))[0]
    self._logits = tf.get_collection_ref(util.with_prefix(self._name, "logits"))[0]
    self._tar_test = tf.get_collection_ref(util.with_prefix(self._name, "tar_test"))[0]
    self._wei_test = tf.get_collection_ref(util.with_prefix(self._name, "wei_test"))[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1


  @property
  def input(self):
    return self._input


  @property
  def cost(self):
    return self._cost
 
  @property
  def logits(self):
    return self._logits

  @property
  def tranM(self):
    return self._tranM

  @property
  def tar_test(self):
    return self._tar_test

  @property
  def wei_test(self):
    return self._wei_test

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name



class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 30
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  tag_size = 6
  rnn_mode = BLOCK


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  tag_size = 6
  rnn_mode = BLOCK


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


def run_epoch(session, model, eval_op = None, verbose=False, test=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  
  fetches = {
      "cost": model.cost,
      "tranM": model.tranM,
      "logits": model.logits,
      "tar_test": model.tar_test,
      "wei_test": model.wei_test
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    
    vals = session.run(fetches, feed_dict)
    
    # train model
    if verbose and step % (model.input.epoch_size // 10) == 10:
      cost = vals["cost"]
      costs += cost
      iters += model.input.num_steps
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))
    # valid & test model, decode
    elif not verbose and step % (model.input.epoch_size // 10) == 10:
      cor_labels = 0
      total_labels = 0

      logits = vals["logits"]
      tranM = vals["tranM"]
      target = vals["tar_test"]
      weight = vals["wei_test"]
      
      for sin_l, sin_t, sin_w in zip(logits, target, weight):
        sin_l = sin_l[:sin_w]
        sin_t = sin_t[:sin_w]
        iterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                sin_l, tranM)
        cor_labels += np.sum(np.equal(iterbi_sequence, sin_t))
        total_labels += sin_w
    
      accuracy = 100 * cor_labels / float(total_labels)
  if verbose:
    return np.exp(costs / iters) 
  else:
    return accuracy
       

def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))
  
  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
   
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)
    
    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)
    
    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)
  
  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f%%" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f%%" % valid_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == "__main__":
  tf.app.run()
