from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np
import random

truncate_size = 100

def read_data(data_path=None):
	train_data=[]	
	label_train_data=[]
	label_train_target=[]
	mask=[];

	#iterate the records
	train_content=file(data_path, 'r').read()
	train_records=train_content.split("\n")
	c_uid=-1
	list_r=[]
	label_list=[]
	target_list=[]
	mask_list=[]
	counter = 0
	for record_index in range(len(train_records)-1):
		columns=train_records[record_index].split(",")
		if(record_index==0):
			c_uid=columns[0]
		if(columns[0]!=c_uid or counter>=truncate_size):
			train_data.append(list_r)
			label_train_data.append(label_list)
			label_train_target.append(target_list)
			mask.append(mask_list)
			list_r=[]
			label_list=[]
			target_list=[]
			mask_list=[];
			c_uid=columns[0]
			counter = 0
		list_r.append(columns[1])
		list_r.append(columns[2])
		list_r.append(columns[3])
		list_r.append(columns[4])
		counter += 1
		label_list.append(columns[1])
		label_list.append(columns[2])
		label_list.append(columns[3])
		label_list.append(columns[4])
		if(len(columns) >= 6):
			target_list.append(columns[5])
			mask_list.append(1)
		else:
			target_list.append(0)
			mask_list.append(0)
	return train_data, label_train_data, label_train_target, mask

def TD_raw_data(data_path=None, data_file=None):
	#set the data path
	if(len(data_file)>0):
		train_path=os.path.join(data_path,"train-"+data_file)
		test_path=os.path.join(data_path,"test-"+data_file)
	else:
		train_path=os.path.join(data_path,"train")
		test_path=os.path.join(data_path,"test")
	
	train_data, label_train_data, label_train_target, train_mask = read_data(data_path=train_path)
	test_data, label_test_data, label_test_target, test_mask = read_data(data_path=test_path)
								
	return train_data, test_data, label_train_data, label_test_data, label_train_target, label_test_target, train_mask, test_mask


class TD_batcher(object):
		#raw_data, raw_seq_lengths=data_padding(raw_data,3)
		def __init__(self, raw_data, raw_label_data, raw_target, mask, batch_size, random_sign=True, name=None):
			"""
			self.raw_data, self.raw_seq_lengths = self.data_padding(raw_data,3)
			self.raw_label_data, self.label_seq_lengths = self.data_padding(raw_label_data, 3)
			self.raw_target, _ = self.data_padding(raw_target, 1)
			self.mask, _ = self.data_padding(mask, 1)
			"""
			self.raw_data=np.array(raw_data);
			self.raw_label_data=np.array(raw_label_data);
			self.raw_target=np.array(raw_target);
			self.mask=np.array(mask);
			self.batch_size = batch_size
			self.epoch_size = len(self.raw_data) // batch_size
			self.batch_idx = 0
			self.random_sign = random_sign

		#padding -1
		def data_padding(self, data, feature_dim):
			num_samples=len(data)
			lengths=[int(len(s)) for s in data]
			max_length=max(lengths)
			padding_dataset=np.full((num_samples, max_length), 0, dtype=np.int)
			for idx, seq in enumerate(data):
				padding_dataset[idx, :len(seq)]=seq
			for idx in range(len(lengths)):
				lengths[idx]=lengths[idx]//feature_dim
			return padding_dataset, lengths

#slicing
		def data_slicing(self, data, feature_dim):
			num_samples=len(data)
			lengths=[int(len(s)) for s in data]
			min_length=min(lengths)
			slicing_dataset=np.full((num_samples, min_length), 0, dtype=np.int)
			for idx, seq in enumerate(data):
				slicing_dataset[idx, :]=seq[:min_length]
			for idx in range(len(lengths)):
				lengths[idx]=lengths[idx]//feature_dim
			return slicing_dataset, lengths
		
		def next_batch(self):
			bs = self.batch_size
			if self.random_sign:
				bid = random.randint(0, self.epoch_size)
			else:
				bid = self.batch_idx
				self.batch_idx = bid + bs			
			raw_data = self.raw_data[bid:bid+bs]
			#raw_seq_lengths = self.raw_seq_lengths[bid:bid+bs]
			raw_data, raw_seq_lengths=self.data_padding(raw_data,4);
			num_tj = len(raw_seq_lengths)
			max_seq_length = int(max(raw_seq_lengths))
			data = np.reshape(raw_data, [num_tj, max_seq_length, 4])
			
			raw_label_data = self.raw_label_data[bid: bid+bs]
			#label_seq_lengths = self.label_seq_lengths[bid: bid+bs]
			raw_label_data, label_seq_lengths=self.data_padding(raw_label_data, 4);
			num_tj = len(label_seq_lengths)
			max_lseq_length = int(max(label_seq_lengths))
			label_data = np.reshape(raw_label_data, [num_tj, max_lseq_length, 4])
		
			raw_target = self.raw_target[bid: bid+bs]
			raw_target, _ = self.data_padding(raw_target, 1);
			target = np.reshape(raw_target, [num_tj, max_lseq_length])

			seq_lengths = raw_seq_lengths
			lseq_lengths = label_seq_lengths
			mask = self.mask[bid:bid+bs]
			mask, _ = self.data_padding(mask, 1);

			x = data 
			y = label_data
			z = target
			x_lens = seq_lengths
			y_lens = lseq_lengths
			return x, y, z, x_lens, y_lens, max_seq_length, max_lseq_length, mask
		