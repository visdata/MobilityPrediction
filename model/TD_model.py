"""
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys

import numpy as np
import tensorflow as tf


def data_type():
	return tf.float32


def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")


# LSTM
class LSTM(object):
	def __init__(self, is_training, config):
		# set the hyperparameters
		batch_size = config.batch_size
		size = config.hidden_size
		vocab_size = config.vocab_size
		# define cell
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)

		if is_training and config.keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
				lstm_cell, output_keep_prob=config.keep_prob)

		if config.num_layers > 1:
			lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

		# initialize state
		self._initial_state = lstm_cell.zero_state(batch_size, data_type())

		# define placeholder
		self.max_seq_len = tf.placeholder(tf.int32, shape=[], name='max_seq_len')
		self.en_input_lens = tf.placeholder(tf.int32, shape=[batch_size], name='seq_lens')
		self.input_hour = tf.placeholder(tf.int32, shape=[batch_size, None], name='hour')
		self.input_minutes = tf.placeholder(tf.int32, shape=[batch_size, None], name='min')
		# self.input_pos = tf.placeholder(tf.int32, shape=[batch_size, None], name='position')
		self.input_lat = tf.placeholder(tf.int32, shape=[batch_size, None], name='position_lat')
		self.input_lon = tf.placeholder(tf.int32, shape=[batch_size, None], name='position_lon')
		self.targets = tf.placeholder(tf.int32, shape=[batch_size, None], name='targets')
		self.mask = tf.placeholder(tf.int32, shape=[batch_size, None], name='mask')
		# embedding
		with tf.device("/cpu:0"):
			self._embedding_hour = tf.get_variable("embedding_hour", [168, size // 4], dtype=data_type())
			inputs_hour = tf.nn.embedding_lookup(self.embedding_hour, self.input_hour)

		with tf.device("/cpu:0"):
			self._embedding_minutes = tf.get_variable("embedding_minutes", [10000, size // 4], dtype=data_type())
			inputs_minutes = tf.nn.embedding_lookup(self.embedding_minutes, self.input_minutes)


		with tf.device("/cpu:0"):
			self._embedding_lat = tf.get_variable("embedding_lat", [1700, size // 4], dtype=data_type(),
												  trainable=True)
			inputs_lat = tf.nn.embedding_lookup(self._embedding_lat, self.input_lat)

		with tf.device("/cpu:0"):
			self._embedding_lon = tf.get_variable("embedding_lon", [2100, size // 4], dtype=data_type(),
												  trainable=True)
			inputs_lon = tf.nn.embedding_lookup(self._embedding_lon, self.input_lon)

		# concat the embedding vector
		inputs = tf.concat([inputs_hour, inputs_minutes, inputs_lat, inputs_lon], 2)

		with tf.variable_scope("RNN"):
			outputs, state = tf.nn.dynamic_rnn(
				lstm_cell, inputs, self.en_input_lens, initial_state=self.initial_state)

		# compute loss
		outputs = tf.reshape(outputs, [-1, size])
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
		output = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
		logits = tf.reshape(output, [-1, vocab_size])
		labels = tf.reshape(self.targets, [-1])
		mask_ = tf.reshape(self.mask, [-1])
		mask_ = tf.cast(mask_, tf.float32)
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[logits],
			[labels],
			[mask_])
		self._cost = cost = tf.reduce_mean(loss)
		self._final_state = state
		self._output = output
		# compute prediction
		predictions = tf.cast(tf.argmax(self.output, 1), tf.int32)
		self._preds = predictions

		# training optimizer
		if not is_training:
			return
		self._lr = tf.Variable(0.1, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
		self._grads = grads
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=tf.contrib.framework.get_or_create_global_step())
		self._new_lr = tf.placeholder(
			tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def output(self):
		return self._output

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def preds(self):
		return self._preds

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def grads(self):
		return self._grads

	@property
	def embedding_hour(self):
		return self._embedding_hour

	@property
	def embedding_minutes(self):
		return self._embedding_minutes

	@property
	def embedding_lat(self):
		return self._embedding_lat

	@property
	def embedding_lon(self):
		return self._embedding_lon

	def get_type(self):
		return 2

# with attention
class ATTModel(object):
	def __init__(self, is_training, config):
		# set the hyperparameters
		batch_size = config.batch_size
		size = config.hidden_size
		vocab_size = config.vocab_size
		# define cell
		lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
		lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
		de_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
		
		if is_training and config.keep_prob < 1:
			lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(
					lstm_cell1, output_keep_prob=config.keep_prob)
			lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
					lstm_cell2, output_keep_prob=config.keep_prob)
			de_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
					de_lstm_cell, output_keep_prob=config.keep_prob)
		if config.num_layers > 1:
			lstm_cell1 = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1] * config.num_layers, state_is_tuple=True)
			lstm_cell2 = tf.nn.rnn_cell.MultiRNNCell([lstm_cell2] * config.num_layers, state_is_tuple=True)

		# initialize state
		self._initial_state = lstm_cell1.zero_state(batch_size, data_type())

		# define placeholder
		self.max_seq_len = tf.placeholder(tf.int32, shape=[], name='max_seq_len')
		self.en_input_lens = tf.placeholder(tf.int32, shape=[batch_size], name='seq_lens')
		self.input_hour = tf.placeholder(tf.int32, shape=[batch_size, None], name='hour')
		self.input_minutes = tf.placeholder(tf.int32, shape=[batch_size, None], name='min')
		#self.input_pos = tf.placeholder(tf.int32, shape=[batch_size, None], name='position')
		self.input_lat = tf.placeholder(tf.int32, shape=[batch_size, None], name='position_lat')
		self.input_lon = tf.placeholder(tf.int32, shape=[batch_size, None], name='position_lon')
		self.max_lseq_len = tf.placeholder(tf.int32, shape=[], name='max_lseq_len')
		self.de_input_lens = tf.placeholder(tf.int32, shape=[batch_size], name='de_seq_lens')
		self.de_input_hour = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_hour')
		self.de_input_minutes = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_min')
		#self.de_input_pos = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_position')
		self.de_input_lat = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_position_lat')
		self.de_input_lon = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_position_lon')
		self.targets = tf.placeholder(tf.int32, shape=[batch_size, None], name='targets')
		self.mask = tf.placeholder(tf.int32, shape=[batch_size, None], name='mask')
		# embedding
		with tf.device("/cpu:0"):
			self._embedding_hour = tf.get_variable("embedding_hour", [168, size // 4], dtype=data_type())
			inputs_hour = tf.nn.embedding_lookup(self.embedding_hour, self.input_hour)
			de_inputs_hour = tf.nn.embedding_lookup(self.embedding_hour, self.de_input_hour)
		with tf.device("/cpu:0"):
			self._embedding_minutes = tf.get_variable("embedding_minutes", [10000, size // 4], dtype=data_type())
			inputs_minutes = tf.nn.embedding_lookup(self.embedding_minutes, self.input_minutes)
			de_inputs_minutes = tf.nn.embedding_lookup(self.embedding_minutes, self.de_input_minutes)
		"""
		with tf.device("/cpu:0"):
			# 33697.3
			self._embedding_pos = tf.get_variable("embedding_pos", [33698, size // 2], dtype=data_type(), trainable=True)
			inputs_pos = tf.nn.embedding_lookup(self._embedding_pos, self.input_pos)
			de_inputs_pos = tf.nn.embedding_lookup(self._embedding_pos, self.de_input_pos)
		"""
		with tf.device("/cpu:0"):
			# 33697.3
			self._embedding_lat = tf.get_variable("embedding_lat", [1700, size // 4], dtype=data_type(),
												  trainable=True)
			inputs_lat = tf.nn.embedding_lookup(self._embedding_lat, self.input_lat)
			de_inputs_lat = tf.nn.embedding_lookup(self._embedding_lat, self.de_input_lat)
		with tf.device("/cpu:0"):
			# 33697.3
			self._embedding_lon = tf.get_variable("embedding_lon", [2100, size // 4], dtype=data_type(),
												  trainable=True)
			inputs_lon = tf.nn.embedding_lookup(self._embedding_lon, self.input_lon)
			de_inputs_lon = tf.nn.embedding_lookup(self._embedding_lon, self.de_input_lon)
		# concat the embedding vector
		inputs = tf.concat([inputs_hour, inputs_minutes, inputs_lat, inputs_lon], 2)
		de_inputs = tf.concat([de_inputs_hour, de_inputs_minutes, de_inputs_lat, de_inputs_lon], 2)

		with tf.variable_scope("RNN"):
			# encoder
			encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
				lstm_cell1, lstm_cell2, inputs, sequence_length=self.en_input_lens, dtype=tf.float32)
			encoder_outputs_fw, encoder_outputs_bw = encoder_outputs
			encoder_state_fw, encoder_state_bw = encoder_state
			self._encoder_state = encoder_state
			# concat foward outputs and backward outputs
			if config.attention == True:
				attention_states = tf.transpose(
					tf.concat((encoder_outputs_fw, encoder_outputs_bw), axis=2),
					[0, 1, 2])
				# attention
				attention_mechanism = tf.contrib.seq2seq.LuongAttention(
					size, attention_states,
					memory_sequence_length=self.en_input_lens)
				de_lstm_cell = tf.contrib.seq2seq.AttentionWrapper(
					de_lstm_cell, attention_mechanism,
					alignment_history=True,
					attention_layer_size=size)
			if config.num_layers > 1:
				de_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([de_lstm_cell] * config.num_layers, state_is_tuple=True)
			# decoder
			# initialize state
			if config.attention == True:
				self._decoder_state = de_lstm_cell.zero_state(batch_size, data_type())
			else:
				self._decoder_state = encoder_state_fw
			decoder_state = self.decoder_state
			# construct the decoder
			fc_layer = tf.contrib.keras.layers.Dense(vocab_size)
			helper = tf.contrib.seq2seq.TrainingHelper(de_inputs, self.de_input_lens)
			decoder = tf.contrib.seq2seq.BasicDecoder(de_lstm_cell, helper, decoder_state, fc_layer)
			outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
			self.alignment_history = state.alignment_history.stack()
			state = state.cell_state
		# compute loss
		logits = tf.reshape(outputs.rnn_output, [-1, vocab_size])
		labels = tf.reshape(self.targets, [-1])
		mask_ = tf.reshape(self.mask, [-1])
		mask_ = tf.cast(mask_, tf.float32)
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[logits],
			[labels],
			[mask_])
		self._cost = cost = tf.reduce_mean(loss)
		self._final_state = state
		self._output = outputs.rnn_output
		# compute prediction
		predictions = tf.cast(tf.argmax(self.output, 2), tf.int32)
		self._preds = predictions

		# training optimizer
		if not is_training:
			return
		self._lr = tf.Variable(0.1, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
		self._grads = grads
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		# optimizer=tf.train.AdagradOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=tf.contrib.framework.get_or_create_global_step())
		self._new_lr = tf.placeholder(
			tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def encoder_state(self):
		return self._encoder_state

	@property
	def decoder_state(self):
		return self._decoder_state

	@property
	def output(self):
		return self._output

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def preds(self):
		return self._preds

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def grads(self):
		return self._grads

	@property
	def embedding_hour(self):
		return self._embedding_hour

	@property
	def embedding_minutes(self):
		return self._embedding_minutes

	@property
	def embedding_lat(self):
		return self._embedding_lat

	@property
	def embedding_lon(self):
		return self._embedding_lon

	def get_type(self):
		return 0

# GRU
class GRU(object):
	def __init__(self, is_training, config):
		# set the hyperparameters
		batch_size = config.batch_size
		size = config.hidden_size
		vocab_size = config.vocab_size
		# define cell
		gru_cell = tf.nn.rnn_cell.GRUCell(size)

		if is_training and config.keep_prob < 1:
			gru_cell = tf.nn.rnn_cell.DropoutWrapper(
				gru_cell, output_keep_prob=config.keep_prob)

		if config.num_layers > 1:
			gru_cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * config.num_layers)

		

		# define placeholder
		self.max_seq_len = tf.placeholder(tf.int32, shape=[], name='max_seq_len')
		self.en_input_lens = tf.placeholder(tf.int32, shape=[batch_size], name='seq_lens')
		self.input_hour = tf.placeholder(tf.int32, shape=[batch_size, None], name='hour')
		self.input_minutes = tf.placeholder(tf.int32, shape=[batch_size, None], name='min')
		# self.input_pos = tf.placeholder(tf.int32, shape=[batch_size, None], name='position')
		self.input_lat = tf.placeholder(tf.int32, shape=[batch_size, None], name='position_lat')
		self.input_lon = tf.placeholder(tf.int32, shape=[batch_size, None], name='position_lon')
		self.targets = tf.placeholder(tf.int32, shape=[batch_size, None], name='targets')
		self.mask = tf.placeholder(tf.int32, shape=[batch_size, None], name='mask')
		# initialize state
		self._initial_state = tf.zeros((batch_size, size), data_type())

		# embedding
		with tf.device("/cpu:0"):
			self._embedding_hour = tf.get_variable("embedding_hour", [168, size // 4], dtype=data_type())
			inputs_hour = tf.nn.embedding_lookup(self.embedding_hour, self.input_hour)

		with tf.device("/cpu:0"):
			self._embedding_minutes = tf.get_variable("embedding_minutes", [10000, size // 4], dtype=data_type())
			inputs_minutes = tf.nn.embedding_lookup(self.embedding_minutes, self.input_minutes)


		with tf.device("/cpu:0"):
			self._embedding_lat = tf.get_variable("embedding_lat", [1700, size // 4], dtype=data_type(),
												  trainable=True)
			inputs_lat = tf.nn.embedding_lookup(self._embedding_lat, self.input_lat)

		with tf.device("/cpu:0"):
			self._embedding_lon = tf.get_variable("embedding_lon", [2100, size // 4], dtype=data_type(),
												  trainable=True)
			inputs_lon = tf.nn.embedding_lookup(self._embedding_lon, self.input_lon)

		# concat the embedding vector
		inputs = tf.concat([inputs_hour, inputs_minutes, inputs_lat, inputs_lon], 2)

		with tf.variable_scope("RNN"):
			outputs, state = tf.nn.dynamic_rnn(
				gru_cell, inputs, self.en_input_lens, initial_state=self.initial_state)
		state = tf.reshape(state, [batch_size, size])

		# compute loss
		outputs = tf.reshape(outputs, [-1, size])
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
		output = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
		logits = tf.reshape(output, [-1, vocab_size])
		labels = tf.reshape(self.targets, [-1])
		mask_ = tf.reshape(self.mask, [-1])
		mask_ = tf.cast(mask_, tf.float32)
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[logits],
			[labels],
			[mask_])
		self._cost = cost = tf.reduce_mean(loss)
		self._final_state = state
		self._output = output
		# compute prediction
		predictions = tf.cast(tf.argmax(self.output, 1), tf.int32)
		self._preds = predictions

		# training optimizer
		if not is_training:
			return
		self._lr = tf.Variable(0.1, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
		self._grads = grads
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=tf.contrib.framework.get_or_create_global_step())
		self._new_lr = tf.placeholder(
			tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def output(self):
		return self._output

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def preds(self):
		return self._preds

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def grads(self):
		return self._grads

	@property
	def embedding_hour(self):
		return self._embedding_hour

	@property
	def embedding_minutes(self):
		return self._embedding_minutes

	@property
	def embedding_lat(self):
		return self._embedding_lat

	@property
	def embedding_lon(self):
		return self._embedding_lon

	def get_type(self):
		return 3


# gru with attention
class GRUATTModel(object):
	def __init__(self, is_training, config):
		# set the hyperparameters
		batch_size = config.batch_size
		size = config.hidden_size
		vocab_size = config.vocab_size
		# define cell
		gru_cell = tf.nn.rnn_cell.GRUCell(size)		
		de_gru_cell = tf.nn.rnn_cell.GRUCell(size)
		
		if is_training and config.keep_prob < 1:
			gru_cell = tf.nn.rnn_cell.DropoutWrapper(
					gru_cell, output_keep_prob=config.keep_prob)
			de_gru_cell = tf.nn.rnn_cell.DropoutWrapper(
					de_gru_cell, output_keep_prob=config.keep_prob)
		if config.num_layers > 1:
			gru_cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell1] * config.num_layers)

		# initialize state
		self._initial_state = tf.zeros((batch_size, size), data_type())

		# define placeholder
		self.max_seq_len = tf.placeholder(tf.int32, shape=[], name='max_seq_len')
		self.en_input_lens = tf.placeholder(tf.int32, shape=[batch_size], name='seq_lens')
		self.input_hour = tf.placeholder(tf.int32, shape=[batch_size, None], name='hour')
		self.input_minutes = tf.placeholder(tf.int32, shape=[batch_size, None], name='min')
		#self.input_pos = tf.placeholder(tf.int32, shape=[batch_size, None], name='position')
		self.input_lat = tf.placeholder(tf.int32, shape=[batch_size, None], name='position_lat')
		self.input_lon = tf.placeholder(tf.int32, shape=[batch_size, None], name='position_lon')
		self.max_lseq_len = tf.placeholder(tf.int32, shape=[], name='max_lseq_len')
		self.de_input_lens = tf.placeholder(tf.int32, shape=[batch_size], name='de_seq_lens')
		self.de_input_hour = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_hour')
		self.de_input_minutes = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_min')
		#self.de_input_pos = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_position')
		self.de_input_lat = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_position_lat')
		self.de_input_lon = tf.placeholder(tf.int32, shape=[batch_size, None], name='dec_position_lon')
		self.targets = tf.placeholder(tf.int32, shape=[batch_size, None], name='targets')
		self.mask = tf.placeholder(tf.int32, shape=[batch_size, None], name='mask')
		# embedding
		with tf.device("/cpu:0"):
			self._embedding_hour = tf.get_variable("embedding_hour", [168, size // 4], dtype=data_type())
			inputs_hour = tf.nn.embedding_lookup(self.embedding_hour, self.input_hour)
			de_inputs_hour = tf.nn.embedding_lookup(self.embedding_hour, self.de_input_hour)
		with tf.device("/cpu:0"):
			self._embedding_minutes = tf.get_variable("embedding_minutes", [10000, size // 4], dtype=data_type())
			inputs_minutes = tf.nn.embedding_lookup(self.embedding_minutes, self.input_minutes)
			de_inputs_minutes = tf.nn.embedding_lookup(self.embedding_minutes, self.de_input_minutes)
		"""
		with tf.device("/cpu:0"):
			# 33697.3
			self._embedding_pos = tf.get_variable("embedding_pos", [33698, size // 2], dtype=data_type(), trainable=True)
			inputs_pos = tf.nn.embedding_lookup(self._embedding_pos, self.input_pos)
			de_inputs_pos = tf.nn.embedding_lookup(self._embedding_pos, self.de_input_pos)
		"""
		with tf.device("/cpu:0"):
			# 33697.3
			self._embedding_lat = tf.get_variable("embedding_lat", [1700, size // 4], dtype=data_type(),
												  trainable=True)
			inputs_lat = tf.nn.embedding_lookup(self._embedding_lat, self.input_lat)
			de_inputs_lat = tf.nn.embedding_lookup(self._embedding_lat, self.de_input_lat)
		with tf.device("/cpu:0"):
			# 33697.3
			self._embedding_lon = tf.get_variable("embedding_lon", [2100, size // 4], dtype=data_type(),
												  trainable=True)
			inputs_lon = tf.nn.embedding_lookup(self._embedding_lon, self.input_lon)
			de_inputs_lon = tf.nn.embedding_lookup(self._embedding_lon, self.de_input_lon)
		# concat the embedding vector
		inputs = tf.concat([inputs_hour, inputs_minutes, inputs_lat, inputs_lon], 2)
		de_inputs = tf.concat([de_inputs_hour, de_inputs_minutes, de_inputs_lat, de_inputs_lon], 2)

		with tf.variable_scope("RNN"):
			# encoder
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
				gru_cell, inputs, self.en_input_lens, initial_state=self.initial_state)
			self._encoder_state = encoder_state
			# concat foward outputs and backward outputs
			if config.attention == True:
				attention_states = tf.transpose(encoder_outputs, [0, 1, 2])
				# attention
				attention_mechanism = tf.contrib.seq2seq.LuongAttention(
					size, attention_states,
					memory_sequence_length=self.en_input_lens)
				de_gru_cell = tf.contrib.seq2seq.AttentionWrapper(
					de_gru_cell, attention_mechanism,
					alignment_history=True,
					attention_layer_size=size)
			if config.num_layers > 1:
				de_gru_cell = tf.nn.rnn_cell.MultiRNNCell([de_gru_cell] * config.num_layers)
			# decoder
			# initialize state
			if config.attention == True:
				self._decoder_state = tf.zeros((batch_size, size), data_type())
			else:
				self._decoder_state = encoder_state
			decoder_state = self.decoder_state
			# construct the decoder
			fc_layer = tf.contrib.keras.layers.Dense(vocab_size)
			helper = tf.contrib.seq2seq.TrainingHelper(de_inputs, self.de_input_lens)
			decoder = tf.contrib.seq2seq.BasicDecoder(de_gru_cell, helper, decoder_state, fc_layer)
			outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
			if config.attention == True:
				self.alignment_history = state.alignment_history.stack()
				state = state.cell_state
			else:
				self.alignment_history = tf.zeros((batch_size, size), data_type())
		# compute loss
		logits = tf.reshape(outputs.rnn_output, [-1, vocab_size])
		labels = tf.reshape(self.targets, [-1])
		mask_ = tf.reshape(self.mask, [-1])
		mask_ = tf.cast(mask_, tf.float32)
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[logits],
			[labels],
			[mask_])
		self._cost = cost = tf.reduce_mean(loss)
		self._final_state = state
		self._output = outputs.rnn_output
		# compute prediction
		predictions = tf.cast(tf.argmax(self.output, 2), tf.int32)
		self._preds = predictions

		# training optimizer
		if not is_training:
			return
		self._lr = tf.Variable(0.1, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
		self._grads = grads
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		# optimizer=tf.train.AdagradOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=tf.contrib.framework.get_or_create_global_step())
		self._new_lr = tf.placeholder(
			tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def encoder_state(self):
		return self._encoder_state

	@property
	def decoder_state(self):
		return self._decoder_state

	@property
	def output(self):
		return self._output

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def preds(self):
		return self._preds

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def grads(self):
		return self._grads

	@property
	def embedding_hour(self):
		return self._embedding_hour

	@property
	def embedding_minutes(self):
		return self._embedding_minutes

	@property
	def embedding_lat(self):
		return self._embedding_lat

	@property
	def embedding_lon(self):
		return self._embedding_lon

	def get_type(self):
		return 4


class trainConfig(object):
	"""train config."""
	init_scale = 0.1
	learning_rate = 0.1
	max_grad_norm = 5
	num_layers = 1
	num_gpus = 1
	num_steps = 200
	hidden_size = 400
	max_epoch = 5
	max_max_epoch = 10
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 2
	attention = False


class testConfig(object):
	"""test config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_gpus = 2
	num_steps = 200
	hidden_size = 400
	max_epoch = 2
	max_max_epoch = 3
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 2
	attention = False
