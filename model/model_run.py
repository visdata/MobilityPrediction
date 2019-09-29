"""
To run:

$ python model_run.py --data_path=/home/tzh/datasets/TD-traindata --data_file=th_15-800_ac --save_path=/home/yurl/jupyter/zhtan/biLSTMModel/ --model biLSTM --mode train

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import tensorflow as tf
import TD_reader
import TD_model

flags = tf.flags
logging = tf.logging

seq_length = 100
results = {
	'input_data': np.array([]).reshape(0, seq_length, 4),
	'input_length': np.array([]),
	'targets': np.array([]).reshape(0, seq_length),
	'predictions': np.array([]).reshape(0, seq_length),
	'alignment_history': np.array([]).reshape(seq_length, 0, seq_length)
}

flags.DEFINE_string(
	"model", None, "A type of model. Possible options are: ATT, biLSTM.")
flags.DEFINE_string("data_path", None,
					"Where the training/test data is stored.")
flags.DEFINE_string("data_file", None,
					"The name of training/test data.")
flags.DEFINE_string("save_path", None,
					"Model output directory.")
flags.DEFINE_string("mode", "train",
					"A type of mode. Possible options are: train, test.")

FLAGS = flags.FLAGS


def data_type():
	return tf.float32


def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")


def load_vec(filename):
	emb = []
	contents = file(filename, 'r').read()
	records = contents.split("\n")
	for index in range(len(records)-1):
		columns = records[index].split(" ")
		emb.append(columns)
	return np.asarray(emb)


# get the performance
def get_performance(output, vocab_size, targets, mask):
	logits = np.reshape(output, [-1, vocab_size])
	labels = np.reshape(targets, [-1])
	mask_ = (np.reshape(mask, [-1])).astype(np.float32)
	# mask_ = np.cast(mask_, np.float32)
	# mask_ = mask_.astype(np.float32)

	predictions = (np.argmax(logits, 1)).astype(np.int32)
	# stayonly_labels = np.scalar_mul(2, labels)
	stayonly_labels = 2 * labels
	travel_tensor = np.ones_like(np.reshape(mask, [-1]))
	stay_tensor = np.zeros_like(np.reshape(mask, [-1]))

	travel_labels = np.equal(labels, travel_tensor)
	stay_labels = np.equal(labels, stay_tensor)
	correct_prediction = np.equal(predictions, labels)
	correct_negative_prediction = np.equal(predictions, stayonly_labels)

	positive = np.sum(travel_labels.astype(np.float32) * mask_)
	negative = np.sum(stay_labels.astype(np.float32) * mask_)
	accuracy = np.sum(correct_prediction.astype(np.float32) * mask_)
	negative_accuracy = np.sum(correct_negative_prediction.astype(np.float32) * mask_)

	TP = accuracy - negative_accuracy
	TN = negative_accuracy
	FN = positive - TP
	FP = negative - TN
	return TP, TN, FN, FP


def run_epoch(session, model, config, batcher, merged_summary_op, epoch_id, log_path=None, eval_op=None, verbose=False, alignment_history_on=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	TPs = 0.0
	TNs = 0.0
	FNs = 0.0
	FPs = 0.0
	state = None
	fetches = {}
	emb_lat_before = None
	emb_lon_before = None
	first_emb = None

	# op to write los to Tensorboard
	summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

	if model.get_type() == 0 or model.get_type() == 2 or model.get_type() == 3:
		state = session.run(model.initial_state)
		fetches = {
			"cost": model.cost,
			"final_state": model.final_state,
			"output": model.output,
			"preds": model.preds,
			"summary": merged_summary_op,
			"emb_lat": model.embedding_lat,
			"emb_lon": model.embedding_lon,
			# "alignment_history": model.alignment_history,
			
		}
	elif model.get_type() == 1:
		fw_state = session.run(model.initial_fw_state)
		bw_state = session.run(model.initial_bw_state)
		fetches = {
			"cost": model.cost,
			"fw_state": model.fw_state,
			"bw_state": model.bw_state,
			"output": model.output,
			"preds": model.preds,
			"summary": merged_summary_op,
			"emb_lat": model.embedding_lat,
			"emb_lon": model.embedding_lon,
			"alignment_history": model.alignment_history,
			
		}
	elif model.get_type() == 4:
		state = session.run(model.initial_state)
		fetches = {
			"cost": model.cost,
			"final_state": model.final_state,
			"output": model.output,
			"preds": model.preds,
			"summary": merged_summary_op,
			"emb_lat": model.embedding_lat,
			"emb_lon": model.embedding_lon,
			"alignment_history": model.alignment_history,	
		}

	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(batcher.epoch_size):
		input_data, de_input, targets, input_lens, de_input_lens, max_seq_len, max_lseq_len, mask = batcher.next_batch()
		
		if alignment_history_on == True:
			results['input_data'] = np.concatenate((results['input_data'], input_data), axis=0)
			results['targets'] = np.concatenate((results['targets'], targets), axis=0)
			results['input_length'] = np.concatenate((results['input_length'], input_lens), axis=0)

		feed_dict = {}
		if model.get_type() == 0 or model.get_type() == 4:
			feed_dict = {model.max_seq_len: max_seq_len, model.input_hour: input_data[:, :, 0],
						 model.input_minutes: input_data[:, :, 1], model.input_lat: input_data[:, :, 2],
						 model.input_lon: input_data[:, :, 3],
						 model.de_input_hour: input_data[:, :, 0], model.de_input_minutes: input_data[:, :, 1],
						 model.de_input_lat: input_data[:, :, 2],  model.de_input_lon: input_data[:, :, 2],
						 model.en_input_lens: input_lens,
						 model.de_input_lens: de_input_lens, model.max_lseq_len: max_lseq_len, model.targets: targets,
						 model.mask: mask, }
		elif model.get_type() == 2:
			feed_dict = {model.max_seq_len: max_seq_len, model.input_hour: input_data[:, :, 0],
						 model.input_minutes: input_data[:, :, 1], model.input_lat: input_data[:, :, 2],
						 model.input_lon: input_data[:, :, 3],model.en_input_lens: input_lens,
						 model.targets: targets,
						 model.mask: mask,
						 model.initial_state.c: state.c, model.initial_state.h: state.h
						 }
		elif model.get_type() == 3:
			feed_dict = {model.max_seq_len: max_seq_len, model.input_hour: input_data[:, :, 0],
						 model.input_minutes: input_data[:, :, 1], model.input_lat: input_data[:, :, 2],
						 model.input_lon: input_data[:, :, 3],model.en_input_lens: input_lens,
						 model.targets: targets,
						 model.mask: mask,
						 model.initial_state: state
						 }


		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		if model.get_type() == 0 or model.get_type == 2:
			state = vals["final_state"]
		elif model.get_type() == 1:
			fw_state = vals["fw_state"]
			bw_state = vals["bw_state"]
		output = vals["output"]
		preds = vals["preds"]
		summary = vals["summary"]
		emb_lat = vals["emb_lat"]
		emb_lon = vals["emb_lon"]
		if alignment_history_on == True:
			alignment_history = vals["alignment_history"]
		
		
		if step==0:
			emb_lat_before = emb_lat
			emb_lon_before = emb_lon
			first_emb = emb_lat

		# write logs
		if log_path:
			summary_writer.add_summary(summary, epoch_id * batcher.epoch_size + step)

		TP, TN, FN, FP = get_performance(output, config.vocab_size, targets, mask)
		costs += cost
		TPs += TP
		TNs += TN
		FNs += FN
		FPs += FP
		iters += 1

		VPs = TPs / (TPs + FPs)  # Travel Precision
		VRs = TPs / (TPs + FNs)  # Travel Recall
		SPs = TNs / (TNs + FNs)  # Stay Precision
		SRs = TNs / (TNs + FPs)  # Stay Recall
		TOL = TPs + TNs + FNs + FPs
		ACC = (TPs + TNs) / TOL

		if verbose and iters % 10 == 0:
			
			print ("Total D-value: ")
			print (np.sum(np.abs(emb_lat - emb_lat_before)) / (emb_lat_before.shape[0] * emb_lat_before.shape[1]))
			print ("Total value: ")
			print (np.sum(np.abs(emb_lat_before)) / (emb_lat_before.shape[0] * emb_lat_before.shape[1]))
			print (np.sum(np.abs(emb_lat)) / (emb_lat.shape[0] * emb_lat.shape[1]))
			print ("ratio: ")
			print (np.sum(np.abs(emb_lat - emb_lat_before)) / np.sum(np.abs(emb_lat_before)))
			
			print("%.3f perplexity: %.3f VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f time: %.0f s" %
				  (
					  iters / batcher.epoch_size, np.exp(costs / iters), VPs, VRs, SPs, SRs, ACC,
					  time.time() - start_time))

		if alignment_history_on == True:
			results['predictions'] = np.concatenate((results['predictions'], preds), axis=0)
			results['alignment_history'] = np.concatenate((results['alignment_history'], alignment_history), axis=1)
		
		emb_lat_before = emb_lat
		emb_lon_before = emb_lon
	
	print ("Total D-value: ")
	print (np.sum(np.abs(emb_lat_before - first_emb)) / (first_emb.shape[0] * first_emb.shape[1]))
	print ("ratio: ")
	print (np.sum(np.abs(emb_lat_before - first_emb)) / np.sum(np.abs(first_emb)))
	

	return np.exp(costs / iters), VPs, VRs, SPs, SRs, ACC, emb_lat_before, emb_lon_before


def get_config():
	if FLAGS.mode == "train":
		return TD_model.trainConfig()
	elif FLAGS.mode == "test":
		return TD_model.trainConfig()
	else:
		raise ValueError("Invalid mode: %s", FLAGS.mode)


def get_model(is_training, config):
	if FLAGS.model == "ATT":
		return TD_model.ATTModel(is_training, config)
	elif FLAGS.model == "biLSTM":
		return TD_model.biLSTMModel(is_training, config)
	elif FLAGS.model == "LSTM":
		return TD_model.LSTM(is_training, config)
	elif FLAGS.model == "ATTModel_emb_offline":
		return TD_model.ATTModel_emb_offline(is_training, config)
	elif FLAGS.model == "GRU":
		return TD_model.GRU(is_training, config)
	elif FLAGS.model == "GRUATT":
		return TD_model.GRUATTModel(is_training, config)
	else:
		raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
	if not FLAGS.data_path:
		raise ValueError("Must set --data_path to TD data directory")

	if not FLAGS.model:
		raise ValueError("Must set --model to TD data directory")

	if not FLAGS.data_file:
		data_file = ""
	else:
		data_file = FLAGS.data_file

	config = get_config()
	test_config = get_config()
	test_config.batch_size = 1000

	path = "./log/" + data_file + "_" + FLAGS.model + "_" + FLAGS.mode
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path + "/summary")
		os.makedirs(path + "/embedding_matrix")
	log_path = path + "/summary"
	log_accuracy_file = path + "/log_accuracy"


	print_to_file(log_accuracy_file, "", "a")

	start_time = time.time()
	raw_data = TD_reader.TD_raw_data(FLAGS.data_path, data_file)
	train_data, test_data, label_train_data, label_test_data, train_target, test_target, train_mask, test_mask = raw_data
	print("data length:" + str(len(train_data)) + ","  + str(len(test_data)) + "," + str(
		len(label_train_data)) + "," + str(len(label_test_data)))


	
	print_to_file(log_accuracy_file, "init_scale: " + str(config.init_scale) + " keep_prob: " + str(config.keep_prob) + " hidden_size: " 
				  + str(config.hidden_size) + " num_layers: " + str(config.num_layers) + " batch_size: " + str(config.batch_size))

	train_summary_list = []
	test_summary_list = []

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
		with tf.variable_scope("Model", reuse=None, initializer=initializer):
			m = get_model(True, config)
		train_summary_list.append(tf.summary.scalar("Training Loss", m.cost))
		train_summary_list.append(tf.summary.scalar("Learning Rate", m.lr))

		with tf.variable_scope("Model", reuse=True, initializer=initializer):
			mtest = get_model(False, test_config)
		test_summary_list.append(tf.summary.scalar("Test Loss", mtest.cost))

		train_merged_summary_op = tf.summary.merge(train_summary_list)
		test_merged_summary_op = tf.summary.merge(test_summary_list)

		sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_summaries_secs=0, save_model_secs=200)
		with sv.managed_session() as session:
			if FLAGS.mode == "train":
				#if FLAGS.model == "ATTModel_emb_offline":
					#m.assign_emb_hour(session, emb_hour_init)
					#m.assign_emb_pos(session, emb_pos_init)
				for i in range(config.max_max_epoch):
					# lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
					lr_decay = 1.0
					m.assign_lr(session, config.learning_rate * lr_decay)
					print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
					print_to_file(log_accuracy_file, "Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
					train_batcher = TD_reader.TD_batcher(train_data, label_train_data, train_target, train_mask,
														 config.batch_size, False)
					train_perplexity, train_VP, train_VR, train_SP, train_SR, train_ACC, emb_lat, emb_lon = run_epoch(
						session, m, config, train_batcher, train_merged_summary_op, i, log_path, eval_op=m.train_op,
						verbose=True)
					F1_S = 2 * train_SP * train_SR / (train_SP + train_SR)
					F1_V = 2 * train_VP * train_VR / (train_VP + train_VR)
					F1 = 2 * F1_S * F1_V / (F1_V + F1_S)
					if i == config.max_max_epoch - 1:
						np.save(path + "/embedding_matrix/emb_lat", emb_lat)
						np.save(path + "/embedding_matrix/emb_lon", emb_lon)
					print(
						"Epoch: %d Train Perplexity: %.3f VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f Time: %.3f" % (
							i + 1, train_perplexity, train_VP, train_VR, train_SP, train_SR, train_ACC, F1_S, F1_V, F1,
							time.time() - start_time))
					print_to_file(log_accuracy_file,
								  "Epoch: %d Train Perplexity: %.3f VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f Time: %.3f" % (
									  i + 1, train_perplexity, train_VP, train_VR, train_SP, train_SR, train_ACC, F1_S, F1_V, F1,
									  time.time() - start_time))

			test_batcher = TD_reader.TD_batcher(test_data, label_test_data, test_target, test_mask,
												test_config.batch_size, False)
			test_perplexity, test_VP, test_VR, test_SP, test_SR, test_ACC, emb_lat, emb_lon = run_epoch(
				session, mtest, test_config, test_batcher, test_merged_summary_op, 0, log_path, None, False, config.attention)

			F1_S = 2 * test_SP * test_SR / (test_SP + test_SR)
			F1_V = 2 * test_VP * test_VR / (test_VP + test_VR)
			F1 = 2 * F1_S * F1_V / (F1_V + F1_S)
			
			print("Epoch: 1 Test Perplexity: %.3f VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f Time: %.3f" % (
				test_perplexity, test_VP, test_VR, test_SP, test_SR, test_ACC, F1_S, F1_V, F1, time.time() - start_time))
			print_to_file(log_accuracy_file,
						  "Epoch: 1 Test Perplexity: %.3f VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f Time: %.3f" % (
							  test_perplexity, test_VP, test_VR, test_SP, test_SR, test_ACC, F1_S, F1_V, F1, time.time() - start_time))

			if FLAGS.save_path and FLAGS.mode == "train":
				print("Saving model to %s." % FLAGS.save_path)
				sv.saver.save(session, FLAGS.save_path + "model", global_step=sv.global_step)

		print("compute time:" + str(time.time() - start_time))
		print_to_file(log_accuracy_file, "compute time:" + str(time.time() - start_time))
		'''
		#np.save('results.npy', results)
		plt.rcParams['figure.figsize'] = (10.0, 10.0) # set default size of plots
		plt.rcParams['image.interpolation'] = 'nearest'
		# plt.rcParams['image.cmap'] = 'gray'
		colors = ['white','red','red'] 
		# bounds = [0,0.5,1]
		# cmap = mpl.colors.ListedColormap(colors)
		cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#FF0000'], 256)
		# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

		input_data = results['input_data']

		labels = results['targets']
		predictions = results['predictions']
		input_length = results['input_length']
		alignment_history = results['alignment_history']
		alignment_history = np.transpose(alignment_history, (1, 0, 2))
		# idx = 7
		
		for idx in xrange(1, 15):
			valid_length = int(input_length[idx])
			trajectory = input_data[idx, 0:valid_length, :].astype('int')
			label = labels[idx, 0:valid_length].astype('int')
			pred = predictions[idx, 0:valid_length].astype('int')
			alignment = alignment_history[idx, 0:valid_length, 0:valid_length].T

			print('valid length:', valid_length)

			plt.imshow(alignment, cmap = cmap)
			# plt.xticks(np.arange(valid_length), pred)
			plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
			plt.xlabel('prediction')
			# plt.yticks(np.arange(valid_length), [str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3]) for i in trajectory])
			plt.yticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
			plt.ylabel('records')
			plt.savefig(path + '/pic/alignment' + str(idx) + '.png')
			plt.show()
			plt.clf()
		
		alignment_average = alignment_history.sum(axis=0) / len(input_length)
		alignment_var = np.zeros(shape=alignment_history[1, : , :].shape)
		for idx in xrange(len(input_length)):
			alignment_var = (alignment_history[idx, :, :] - alignment_average) ** 2
		print("Var: %f" % np.mean(alignment_var))
		plt.imshow(alignment_average.T, cmap = cmap)
		plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
		plt.xlabel('prediction')
		plt.yticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
		plt.ylabel('records')
		plt.savefig(path + '/pic/alignment' + 'alignment_average.png')
		plt.show()
		plt.clf()
		'''

		
		


if __name__ == "__main__":
	tf.app.run()
