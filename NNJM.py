# coding: utf-8

#NNJM

#TODO
# Convolution instead of first dense hidden layer
# Put in character CNN
# Put in bi-dir LSTm
# Add in different loss function
# Add in argparse
# Save models

#Import statements
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import codecs
import types
import sys

#Process the data, get the source-vocab and target-vocab variables
import data_loader

#Parameters for the model
par = {}
par['minibatch'] = 64
par['in-emb-size'] = 50
par['out-emb-size'] = 256
par['hs-size'] = 512
par['src-voc'] = None
par['tgt-voc'] = None
par['dt'] = tf.float32
par['src-win'] =11
par['tgt-win'] = 4
par['char'] = False
par['emb-dropout-rate'] = 0.1 #probability of dropping a node
par['l1-dropout-rate'] = 0.1
par['l2-dropout-rate'] = 0.4
par['loss'] = 'MLE' #('MLE','NCE','IS')
par['lr'] = 0.8 #learning rate
par['NCE-samples'] = 100
par['dec-fac'] = 0.5 #Multiply the learning rate by this if dev perplexity increases by more than epsilon
par['ep-crit'] = 0 #Epsilon in decrease-factor
par['val-check-rate'] = 0.5 #every fraction of the training set get val perplexity
par['epochs'] = 100 #how many epochs to train for
par['mb-per-ep'] = None #Filled in by the data preprep
par['src-train-file'] = 'train.source.lc'
par['tgt-train-file'] = 'train.target.lc'
par['mapping-file-name'] = 'mapping.nn'
par['custom-target-words'] = 'output.words' #This forces the input and output target words to use the words specified in this list
par['count-cutoff'] = 3
par['training-data-file-name'] = 'training.data.11+4'
par['val-data-file-name'] = 'validation.data.11+4'
par['gpu'] = True

#Character model params
_char_params = {}
_char_params['num-highway-layers'] = 4
_char_params['char-emb-size'] = 25
_char_params['filter-width'] = 7
_char_params['longest-word'] = 30

_init_params = {}
_init_params['init-meth'] = 'uniform'
_init_params['init-range'] = 0.01
if _init_params['init-meth'] == 'uniform':
    _init_params['di'] = tf.random_uniform_initializer(minval=-1*_init_params['init-range'], maxval=_init_params['init-range'],seed=None, dtype=par['dt'])

par['char-par'] = _char_params
par['init-par'] = _init_params


#Error checking for user inputs
class NNJM_Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
try:
    if not par['val-check-rate']>0: raise NNJM_Error('val-check-rate')
    if not par['val-check-rate']<=1: raise NNJM_Error('val-check-rate')
    if not par['emb-dropout-rate']>=0: raise NNJM_Error('dropout-rate')
    if not par['emb-dropout-rate']<=1: raise NNJM_Error('dropout-rate')
    if not par['l1-dropout-rate']>=0: raise NNJM_Error('dropout-rate')
    if not par['l1-dropout-rate']>=0: raise NNJM_Error('dropout-rate')
    if not par['l2-dropout-rate']<=1: raise NNJM_Error('dropout-rate')
    if not par['l2-dropout-rate']<=1: raise NNJM_Error('dropout-rate')
    if not par['init-par']['init-range']>0: raise NNJM_Error('init-range')
    if not par['minibatch']>0: raise NNJM_Error('minibatch')
    if not par['lr']>=0: raise NNJM_Error('learning-rate')
    if not type(par['gpu'])==types.BooleanType: raise NNJM_Error('gpu')
    if not type(par['char'])==types.BooleanType: raise NNJM_Error('use-char')
    if not par['loss'] in ['MLE','NCE','IS']: raise NNJM_Error('loss')
except NNJM_Error:
    print("Input Error",e.value)

#Interactive session
sess = tf.Session() 

#Process the data, get the source-vocab and target-vocab variables
import data_loader

#create the mapping file
#print('*'*10,"WARNING A NEW MAPPING FILE IS NOT BEING CREATED, UNCOMMENT THE LINE IN THE CODE IF YOU WANT TO GENERATE NEW ONES",'*'*10)
data_loader.create_word_mapping_file(par['src-train-file'],par['tgt-train-file'],par['mapping-file-name'],par['count-cutoff'],par['custom-target-words'])

#Now load in the data
data_factory = data_loader.minibatcher(par['mapping-file-name'],par['training-data-file-name'],par['val-data-file-name'],par['src-win'],par['tgt-win'],par['minibatch'],par['val-check-rate'])
par['mb-per-epoch'] = data_factory.minibatches_per_epoch()
par['src-voc'] = data_factory.source_vocab_size
par['tgt-voc'] = data_factory.target_vocab_size


#Place holders for input and output data, first index is the minibatch size

#For the input the second dimension will be passed a
#     vector of size minibatch x (par['src-win']+par['tgt-win'])
with tf.device('/gpu:0' if par['gpu'] else '/cpu:0'):
    input_indices = tf.placeholder(tf.int64, shape=[None, par['src-win']+par['tgt-win']],name='input_indices')

#The output will be given a minibatch vector of correct indicies
with tf.device('/gpu:0' if par['gpu'] else '/cpu:0'):
    correct_output = tf.placeholder(tf.int64,shape=[None],name='correct_output')

#For passing in the learning rate and dropout rate
with tf.device('/gpu:0' if par['gpu'] else '/cpu:0'):
    learning_rate = tf.placeholder(par['dt'],shape=[],name='learning-rate')
    emb_dropout_rate = tf.placeholder(par['dt'],shape=[],name='emb-dropout-rate')
    l1_dropout_rate = tf.placeholder(par['dt'],shape=[],name='l1-dropout-rate')
    l2_dropout_rate = tf.placeholder(par['dt'],shape=[],name='l2-dropout-rate')


def _variable(name,shape,initializer,use_gpu):
    assert shape != None,"Error shape cannot be None in _variable"
    assert type(use_gpu)==types.BooleanType
    with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var

#Parameters for the source and target embeddings
with tf.variable_scope('input'):
    source_emb_matrix = _variable('source-emb-matrix', [par['src-voc'], par['in-emb-size']],par['init-par']['di'],par['gpu'])
    target_emb_matrix = _variable('target-emb-matrix',[par['tgt-voc'],par['in-emb-size']],par['init-par']['di'],par['gpu']) 
#Emebedding lookups
src_embed = tf.nn.embedding_lookup(source_emb_matrix, tf.slice(input_indices,[0,0],[-1,par['src-win']]))
tgt_embed = tf.nn.embedding_lookup(target_emb_matrix, tf.slice(input_indices,[0,par['src-win']],[-1,par['tgt-win']]))

#Now reshape to be able to feed through non-linearity
concat_embed = tf.concat(1, [src_embed, tgt_embed])
concat_embed_reshape = tf.reshape(concat_embed,[-1,par['in-emb-size']*(par['src-win']+par['tgt-win'])])
concat_embed_do = tf.nn.dropout(concat_embed_reshape, 1-emb_dropout_rate) #Pass in the keep prob

#First Layer
with tf.variable_scope('layer1'):
    weights = _variable('weights',[par['in-emb-size']*(par['src-win']+par['tgt-win']),par['hs-size']],par['init-par']['di'],par['gpu'])
    bias = _variable('bias',[par['hs-size']],par['init-par']['di'],par['gpu'])
    output = tf.nn.relu(tf.matmul(concat_embed_do,weights)+bias)
    layer1 = tf.nn.dropout(output, 1-l1_dropout_rate) #Pass in the keep prob

#Second layer
with tf.variable_scope('layer2'):
    weights = _variable('weights',[par['hs-size'],par['out-emb-size']],par['init-par']['di'],par['gpu'])
    bias = _variable('bias',[par['out-emb-size']],par['init-par']['di'],par['gpu'])
    output = tf.nn.relu(tf.matmul(layer1,weights)+bias)
    layer2 = tf.nn.dropout(output, 1-l2_dropout_rate) #Pass in the keep prob


#Softmax layer
with tf.variable_scope('output'):
    di = par['init-par']['di']
    weights = _variable('weights',[par['out-emb-size'],par['tgt-voc']],di,par['gpu'])
    bias = None
    #Is using NCE initialize bias to -log(V)
    if par['loss'] == 'NCE':
        ci = tf.constant_initializer(value=-np.log(par['tgt-voc']), dtype=par['dt'])
        bias = _variable('bias',[par['tgt-voc']],ci,par['gpu'])
    else: 
        bias = _variable('bias',[par['tgt-voc']],di,par['gpu'])
	unscaled_final_output = tf.matmul(layer2, weights) + bias
    loss = None
    if par['loss'] == 'MLE':
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_final_output,correct_output))
    elif par['loss'] == 'NCE':
        loss = tf.reduce_mean(tf.nn.nce_loss(tf.transpose(weights), bias, unscaled_final_output, correct_output, par['num-NCE-samples'], par['tgt-voc']))
    m_log_prob = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_final_output, correct_output)) #used for perp

#Optimizer
optim = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optim.compute_gradients(loss)
scaled_grads_and_vars = [(tf.mul(gv[0],1.0/par['minibatch']),gv[1]) for gv in grads_and_vars]
train_op = optim.apply_gradients(scaled_grads_and_vars)

#Initiailize all the variables
sess.run(tf.initialize_all_variables())

#Additional variables for tracking loss
val_perplexities = [] #stores the validation perplexities each validation point


################# DEBUGGING ##################


#print('source-emb-matrix:\n',source_emb_matrix.eval(session=sess),source_emb_matrix.get_shape()) 
#print('target-emb-matrix:\n',target_emb_matrix.eval(session=sess),target_emb_matrix.get_shape()) 
#data_factory.prep_train()
#curr_minibatch,eval_val = data_factory.get_minibatch()
#print('minibatch:\n',curr_minibatch)
#input_train_batch = curr_minibatch[:,:par['src-win']+par['tgt-win']]
#output_train_batch = np.squeeze(np.copy(curr_minibatch[:,par['src-win']+par['tgt-win']:]))
#print('input minibatch:\n',input_train_batch)
#print('output minibatch:\n',output_train_batch)
#print('source slice:\n',tf.slice(input_indices,[0,0],[-1,par['src-win']]).eval(session=sess,feed_dict={input_indices:input_train_batch,
#                correct_output:output_train_batch}))
#print('target slice:\n',tf.slice(input_indices,[0,par['src-win']],[-1,par['tgt-win']]).eval(session=sess,feed_dict={input_indices:input_train_batch}))
#
#print('lookedup embeddings source:\n',src_embed.eval(session=sess,feed_dict={input_indices:input_train_batch}))
#print('lookedup embeddings target:\n',tgt_embed.eval(session=sess,feed_dict={input_indices:input_train_batch}))
#print('concat embed:\n',concat_embed.eval(session=sess,feed_dict={input_indices:input_train_batch}))
#print('concat embed reshape:\n',concat_embed_reshape.eval(session=sess,feed_dict={input_indices:input_train_batch}))
#print('concat embed do:\n',concat_embed_do.eval(session=sess,feed_dict={input_indices:input_train_batch,dropout_rate:0.1}))
#print('unint minibatch:\n',data_factory.unint_batch(curr_minibatch))
#sys.exit()
##############################################


#Do all the training
print("Training for",par['epochs'],"epochs (",par['epochs']*par['mb-per-epoch'],"minibatches)")
print("All current parameters:")
print(par,'\n\n')
print("All trainable variables in the model")
print("-"*10,' All trainable variables in model (name,dtype,shape,device) ','-'*10)
for var in tf.all_variables():
    print(var.name,var.dtype,var.get_shape(),var.device)

print("-"*10,'beginning training','-'*10)

start_train = time.time()
data_factory.prep_train() #prep the data loader for training, also stores timing info
for i in range(par['epochs']*par['mb-per-epoch']):
    #get the minibatch
    curr_minibatch,eval_val = data_factory.get_minibatch()
    if eval_val:
        start_val = time.time()
        log_sum = 0
        total_words = 0
        for val_batch in data_factory.get_val_data_gen():
            #print val_batch[:,:par['src-win']+par['tgt-win']].shape
            #print np.squeeze(val_batch[:,par['src-win']+par['tgt-win']:]).shape
            input_val_batch = val_batch[:,:par['src-win']+par['tgt-win']]
            output_val_batch = np.squeeze(np.copy(val_batch[:,par['src-win']+par['tgt-win']:]))
            log_sum+=m_log_prob.eval(session=sess,feed_dict={input_indices:input_val_batch,
                        correct_output:output_val_batch, 
                        l1_dropout_rate:0.0,l2_dropout_rate:0.0,emb_dropout_rate:0.0})
            total_words+=val_batch.shape[0]
        log_sum = (log_sum/np.log(2.0))/total_words
        print("Perplexity on validation set:",2**log_sum)
        val_perplexities.append(2**log_sum)
        if (len(val_perplexities) > 1) and (val_perplexities[-2] + par['ep-crit'] < val_perplexities[-1]):
            par['lr']*=par['dec-fac']
            print('Decreased learning rate to:',par['lr'])
        end_val = time.time()
        print("Time for perplexity on dev set (minutes):",(end_val - start_val)/60.0)
    #Now update the gradients for this training batch
    input_train_batch = curr_minibatch[:,:par['src-win']+par['tgt-win']]
    output_train_batch = np.squeeze(np.copy(curr_minibatch[:,par['src-win']+par['tgt-win']:]))
    train_op.run(session=sess,feed_dict={input_indices:input_train_batch,
                correct_output:output_train_batch,\
                emb_dropout_rate:par['emb-dropout-rate'],
                l1_dropout_rate:par['l1-dropout-rate'],
                l2_dropout_rate:par['l2-dropout-rate'],
                learning_rate:par['lr'] })

end_train = time.time()
print("Time for total training (minutes):",(end_train - start_train)/60.0)

