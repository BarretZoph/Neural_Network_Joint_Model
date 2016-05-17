import codecs
from collections import defaultdict as dd
import sys
import itertools
import random
import numpy as np
import os

def create_word_mapping_file(src_train_file,tgt_train_file,mapping_name,count_cutoff):
	print "source file name used for creating mapping file:",src_train_file
	print "target file name used for creating mapping file:",tgt_train_file	
	src_train_file = codecs.open(src_train_file,'r','utf-8')
	tgt_train_file = codecs.open(tgt_train_file,'r','utf-8')	
	src_counts = dd(int)
	tgt_counts = dd(int)
	for src_line,tgt_line in itertools.izip(src_train_file,tgt_train_file):
		src_line = src_line.replace('\n','').split(' ')
		tgt_line = tgt_line.replace('\n','').split(' ')
		for word in src_line:
			src_counts[word]+=1
		for word in tgt_line:
			tgt_counts[word]+=1
	if '' in src_counts: del src_counts['']
	if '' in tgt_counts: del tgt_counts['']
	src_words = [x for x in src_counts if src_counts[x] >= count_cutoff]
	tgt_words = [x for x in tgt_counts if tgt_counts[x] >= count_cutoff]
	src_vocab_mapping = {}
	tgt_vocab_mapping = {}
	for idx,word in enumerate(src_words):
		src_vocab_mapping[idx] = word
	for idx,word in enumerate(tgt_words):
		tgt_vocab_mapping[idx] = word
	print 'source vocab size:',len(src_vocab_mapping)+3
	print 'target vocab size:',len(tgt_vocab_mapping)+3
	mapping_file = codecs.open(mapping_name,'w','utf-8')
	mapping_file.write(str(0)+'\t'+'<unk>\n')
	mapping_file.write(str(1)+'\t'+'<source_s>\n')
	mapping_file.write(str(2)+'\t'+'<\source_s>\n')
	for i in range(0,len(src_vocab_mapping)):
		mapping_file.write(str(i+3)+'\t'+src_vocab_mapping[i]+'\n')
	mapping_file.write('='*10+'\n')
	mapping_file.write(str(0)+'\t'+'<unk>\n')
	mapping_file.write(str(1)+'\t'+'<s>\n')
	mapping_file.write(str(2)+'\t'+'<\s>\n')
	for i in range(0,len(tgt_vocab_mapping)):
		mapping_file.write(str(i+3)+'\t'+tgt_vocab_mapping[i]+'\n')

class minibatcher:
	def __init__(self,mapping_file,data_file_name,val_file_name,src_window,tgt_window,minibatch_size,val_check_rate):
		self.load_mapping(mapping_file)
		self.src_window = src_window
		self.tgt_window = tgt_window
		self.current_index = 0
		self.minibatch_size = minibatch_size
		print "src_window:",self.src_window
		print "tgt_window:",self.tgt_window
		print "minibatch_size:",self.minibatch_size
		self.load_data(data_file_name,val_file_name)
		#now set the checkpoints for getting val score
		self.val_checkpoints = []
		self.current_epoch = 1
		tmp_val = 0.0
		while tmp_val <1:
			self.val_checkpoints.append(int(tmp_val*self.data.shape[0]))
			tmp_val+=val_check_rate
		print "Number of times per epoch the validation set will be evaluated:",len(self.val_checkpoints)			
	def load_mapping(self,mapping_file):
		source = True
		self.src_mapping = {}
		self.tgt_mapping = {}
		for line in codecs.open(mapping_file,'r','utf-8'):
			if line[0] == '=':
				source = False
				continue
			line = line.replace('\n','').split('\t')
			if source:
				self.src_mapping[line[1]] = int(line[0])
			else:
				self.tgt_mapping[line[1]] = int(line[0])
		self.source_vocab_size = len(self.src_mapping)
		self.target_vocab_size = len(self.tgt_mapping)
		print "Source mapping size:",len(self.src_mapping)
		print "Target mapping size:",len(self.tgt_mapping)
	def load_data(self,data_file_name,val_file_name):
		self.data = []
		self.val_data = []
		for line in codecs.open(data_file_name,'r','utf-8'):
			line = line.replace('\n','').split(' ')
			assert len(line)==(self.src_window+self.tgt_window+1),"Error not correct data format"
			src_words = line[:self.src_window]
			tgt_words = line[self.src_window:]	
			for idx,word in enumerate(src_words):
				assert word!='<unk>',"Error the dataset has already been unked"
				if word in self.src_mapping:
					src_words[idx] = self.src_mapping[word]
				else:
					src_words[idx] = self.src_mapping['<unk>']
			for idx,word in enumerate(tgt_words):
				assert word!='<unk',"Error the dataset has already been unked"
				if word in self.tgt_mapping:
					tgt_words[idx] = self.tgt_mapping[word]
				else:
					tgt_words[idx] = self.tgt_mapping['<unk>']
			
			self.data.append(src_words+tgt_words)
		random.shuffle(self.data)
		self.data = np.matrix(self.data)
		assert self.data.shape[0]>=self.minibatch_size,"Error length of the data is less than the minibatch size"
		print "Number of training examples:",self.data.shape[0]
		#Now start the validation data preprocessing
		for line in codecs.open(val_file_name,'r','utf-8'):
			line = line.replace('\n','').split(' ')
			assert len(line)==(self.src_window+self.tgt_window+1),"Error not correct data format"
			src_words = line[:self.src_window]
			tgt_words = line[self.src_window:]	
			for idx,word in enumerate(src_words):
				assert word!='<unk>',"Error the dataset has already been unked"
				if word in self.src_mapping:
					src_words[idx] = self.src_mapping[word]
				else:
					src_words[idx] = self.src_mapping['<unk>']
			for idx,word in enumerate(tgt_words):
				assert word!='<unk',"Error the dataset has already been unked"
				if word in self.tgt_mapping:
					tgt_words[idx] = self.tgt_mapping[word]
				else:
					tgt_words[idx] = self.tgt_mapping['<unk>']
			self.val_data.append(src_words+tgt_words)
		self.val_data = np.matrix(self.val_data)
		self.val_stride = min(32,self.val_data.shape[0])
		print "val stride:",self.val_stride
		print "val shape:",self.val_data.shape
	def get_minibatch(self):
		final_data = None
		score_val = False #Should we score the validation after this minibatch
		if self.within_range():
			score_val = True	
		if self.current_index + self.minibatch_size > self.data.shape[0]:
			#Now we need to wrap around the training set
			print '-'*10,"Epoch",self.current_epoch,"just finished",'-'*10
			self.current_epoch+=1
			bottom_data = self.data[self.current_index:,:]
			self.current_index = self.minibatch_size - (self.data.shape[0] - self.current_index)
			top_data = self.data[:self.current_index]
			final_data = np.concatenate((bottom_data, top_data), axis=0)
		else:
			final_data = self.data[self.current_index:self.current_index+self.minibatch_size,:]
			self.current_index = self.current_index + self.minibatch_size
		return final_data,score_val
	def within_range(self):
		for num in self.val_checkpoints:
			if self.current_index>=num and self.current_index-self.minibatch_size<num:
				return True
	def minibatches_per_epoch(self):
		tmp_num = self.data.shape[0]/self.minibatch_size
		if self.data.shape[0]%self.minibatch_size != 0:
			tmp_num+=1
		return tmp_num
	def get_val_data_gen(self):
		num_steps = None	
		if self.val_stride != 32: #This means the data is less than the stride index, so just use whole data
			nume_steps = 1	
		else:
			num_steps = self.val_data.shape[0]/self.val_stride
			if self.val_data.shape[0]%self.val_stride != 0:
				num_steps +=1	
		for i in range(0,num_steps-1):
			curr_index = i*self.val_stride
			yield self.val_data[curr_index:curr_index+self.val_stride]
		yield self.val_data[(num_steps-1)*self.val_stride:]
		
	
##Now begins the testing
##np.random.seed(seed=1)
###create_word_mapping_file(sys.argv[1],sys.argv[2],'mapping.nn',3)
##mybatcher = minibatcher('mapping.nn','training.data.11+4.small','validation.data.11+4',11,4,20,0.5)
##
##print "Minibatches per epoch:",mybatcher.minibatches_per_epoch()
##for batch in mybatcher.get_val_data_gen():
##	print batch.shape
#
