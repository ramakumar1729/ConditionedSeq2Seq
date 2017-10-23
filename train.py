# Training script for Zero Shot Slot Tagging models.

import numpy as np
import tensorflow as tf
from tensorflow import gfile
# TODO(ramakumar): use estimator.train/predict/evaluate instead of learn runner. 
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.experiment  import Experiment

from . import input_fn
from . import seq2seq_model

tf.flags.DEFINE_string("schedule", "continuous_train_and_eval",
                       """Estimator function to call, defaults to
                       continuous_train_and_eval for local run""")
tf.flags.DEFINE_string("output_dir", "./output_dir",  """The directory to write model checkpoints and summaries
                       to. If None, a local temporary directory is created.""")
tf.flags.DEFINE_string("train_file", "", """train TFRecord file.""")
tf.flags.DEFINE_string("dev_file", "", """dev TFRecord file.""")
tf.flags.DEFINE_string("test_file", "", """test TFRecord file.""")
tf.flags.DEFINE_string("source_vocab_file","" , """source vocab file.""")
tf.flags.DEFINE_string("target_vocab_file","" , """target vocab file.""")
tf.flags.DEFINE_string("tag_vocab_file","" , """tag vocab file.""")

FLAGS = tf.flags.FLAGS

def create_experiment(output_dir):
  
  config = run_config.RunConfig(
      tf_random_seed=FLAGS.tf_random_seed,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      save_checkpoints_steps=1000,
      keep_checkpoint_max=5,
      keep_checkpoint_every_n_hours=4,
      gpu_memory_fraction=1.0)
  config.tf_config.gpu_options.allow_growth = True
  config.tf_config.log_device_placement = True

  # TODO(ramakumar): Create train/dev/test splits and vocab files.
  train_file = FLAGS.train_file
  dev_file = FLAGS.dev_file
  source_vocab_file = FLAGS.source_vocab_file
  target_vocab_file = FLAGS.target_vocab_file
  tag_vocab_file = FLAGS.tag_vocab_file
  train_input_fn = input_fn.input_fn(train_file, source_vocab_file, target_vocab_file, tag_vocab_file) 
  eval_input_fn = input_fn.input_fn(dev_file, source_vocab_file, target_vocab_file, tag_vocab_file) 

  # TODO(ramakumar): Create seq2seq_model_fn.
  model_fn = seq2seq_model.seq2seq_model_fn

  # Can update model params here.
  model_params = {} # Using default params.
  estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

  experiment = Experiment(
                  estimator=estimator,
                  train_input_fn=train_input_fn,
                  eval_input_fn=eval_input_fn,
                  min_eval_frequency=1000,
                  train_steps=10000)

  return experiment


def main(_):
  gfile.MakeDirs(output_dir)
  learn_runner.run(
    experiment_fn=create_experiment,
    output_dir=FLAGS.output_dir,
    schedule=FLAGS.schedule
    )

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
