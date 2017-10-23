# Creates input_fn from TFRecords of source, target and tag_id.

import numpy as np
import tensorflow as tf

from . import vocab

# Importing data using tf.train.dataset.
# https://www.tensorflow.org/programmers_guide/datasets.
def create_input_fn(filename, vocab_source, vocab_target, vocab_tag):
  dataset = tf.contrib.data.TFRecordDataset(filename)
  
  PREPEND_TOKEN = "SEQUENCE_START"
  APPEND_TOKEN = "SEQUENCE_END"

  # Create vocabulary lookup for source.
  source_vocab_to_id, source_id_to_vocab, source_word_to_count, _ = \
    vocab.create_vocabulary_lookup_table(vocab_source)

  # Create vocabulary look for target.
  target_vocab_to_id, target_id_to_vocab, target_word_to_count, _ = \
    vocab.create_vocabulary_lookup_table(vocab_target)
  
  # Create vocab for tags.
  tag_vocab_to_id, tag_id_to_vocab, tag_word_to_count, _ = \
    vocab.create_vocabulary_lookup_table(vocab_tag)

  def parser(record):
    key_to_features = {
      "tag" : tf.FixedLenFeature((), tf.int64, default_value=""),
      "source" : tf.FixedLenFeature((), tf.string),
      "target" : tf.FixedLenFeature((), tf.string, default_value="")
      }
    parsed = tf.parse_single_example(record, keys_to_features)

    source, target, tag = parsed["source"], parsed["target"], parsed["tag"]

    # Split source tokens and add SOS, EOS.
    source_tokens = tf.string_split([source], delimiter=" ").values
    source_tokens = tf.concat([[PREPEND_TOKEN], tokens, [APPEND_TOKEN]], 0)
    source_ids = source_vocab_to_id.lookup(source_tokens)
    source_len = tf.size(source_tokens)

    # Split target tokens and add SOS, EOS.
    target_tokens = tf.string_split([target], delimiter=" ").values
    target_tokens = tf.concat([[PREPEND_TOKEN], tokens, [APPEND_TOKEN]], 0)
    target_ids = target_vocab_to_id.lookup(target_tokens)
    target_len = tf.size(target_tokens)

    tag_id = tag_vocab_to_id.lookup(tag)

    return ((source_ids, source_len), (target_ids, target_len), tag_id)

  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  # dataset = dataset.batch(32)
  # Using padded_batch.
  dataset = dataset.padded_batch(
                  32,
                  (tf.TensorShape([None]), # source vectors of unknown size
                  tf.TensorShape([])),     # size(source)
                  (tf.TensorShape([None]), # target vectors of unknown size
                  tf.TensorShape([])),     # size(target)
                  tf.TensorShape([])      # tag_id
                  )
  dataset = dataset.repeat(10)
  iterator = dataset.make_one_shot_iterator()

  ((source_ids, source_len), (target_id, target_len)) = iterator.get_next()
  features = { "source_ids": source_ids, "source_len": source_len, "tag_id": tag_id}
  labels = { "target_ids" : target_ids, "target_len": target_len}
  return features, labels
  
