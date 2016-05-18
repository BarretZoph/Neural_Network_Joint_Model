
# coding: utf-8

# In[1]:

#NNJM
#TODO
# Check to be sure slices are being done correctly
# Convolution instead of first dense hidden layer
# Put in character CNN
# Put in bi-dir LSTm
# Add in different loss function
# Add in argparse
# Save models


# In[2]:

#Import statements
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import codecs
import types
get_ipython().magic(u'matplotlib inline')


# In[3]:

#Parameters for the model
params = {}
params['minibatch'] = 64
params['input-emb-size'] = 100
params['output-emb-size'] = 100
params['hiddenstate-size'] = 400
params['source-vocab'] = None
params['target-vocab'] = None
params['datatype'] = tf.float32
params['init-method'] = 'uniform'
params['init-range'] = 0.01
params['source-window'] = 11
params['target-window'] = 4
params['use-char'] = False
params['seed'] = 1
params['dropout-rate'] = 0.0 #probability of dropping a node
params['loss'] = 'MLE' #('MLE','NCE','IS')
params['learning-rate'] = 0.1
params['decrease-factor'] = 0.8 #Multiply the learning rate by this if dev perplexity increases by more than epsilon
params['epsilon-criteria'] = 0 #Epsilon in decrease-factor
params['val-check-rate'] = 0.5 #every fraction of the training set get val perplexity
params['epochs'] = 1000 #how many epochs to train for
params['minibatches-per-epoch'] = None #Filled in by the data preprep
params['source-train-file'] = 'train.source.lc'
params['target-train-file'] = 'train.target.lc'
params['mapping-file-name'] = 'mapping.nn'
params['count-cutoff'] = 3
params['training-data-file-name'] = 'training.data.11+4.smaller'
params['val-data-file-name'] = 'training.data.11+4.smaller'#'validation.data.11+4'
params['use-gpu'] = False

#Character model params
char_params = {}
char_params['num-highway-layers'] = 4
char_params['char-emb-size'] = 25
char_params['filter-width'] = 7
char_params['longest-word'] = 30


params['char-params'] = char_params


# In[4]:

#Error checking for user inputs
class NNJM_Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
try:
    if not params['val-check-rate']>0: raise NNJM_Error('val-check-rate')
    if not params['val-check-rate']<=1: raise NNJM_Error('val-check-rate')
    if not params['dropout-rate']>=0: raise NNJM_Error('dropout-rate')
    if not params['dropout-rate']<=1: raise NNJM_Error('dropout-rate')
    if not params['init-range']>0: raise NNJM_Error('init-range')
    if not params['minibatch']>0: raise NNJM_Error('minibatch')
    if not params['learning-rate']>=0: raise NNJM_Error('learning-rate')
    if not type(params['use-gpu'])==types.BooleanType: raise NNJM_Error('use-gpu')
    if not type(params['use-char'])==types.BooleanType: raise NNJM_Error('use-char')
    if not params['loss'] in ['MLE','NCE','IS']: raise NNJM_Error('loss')
except NNJM_Error as e:
        print 'Bad user input was entered for:', e.value


# In[5]:

#Interactive session
np.random.seed(seed=params['seed'])
sess = tf.InteractiveSession()


# In[6]:

#Process the data, get the source-vocab and target-vocab variables
import data_loader


# In[7]:

#create the mapping file
data_loader.create_word_mapping_file(params['source-train-file'],params['target-train-file'],            params['mapping-file-name'],params['count-cutoff'])


# In[8]:

#Now load in the data
data_factory = data_loader.minibatcher(params['mapping-file-name'],params['training-data-file-name'],                params['val-data-file-name'],params['source-window'],params['target-window'],                params['minibatch'],params['val-check-rate'])
params['minibatches-per-epoch'] = data_factory.minibatches_per_epoch()
params['source-vocab'] = data_factory.source_vocab_size
params['target-vocab'] = data_factory.target_vocab_size


# In[9]:

#Place holders for input and output data, first index is the minibatch size

#For the input the second dimension will be passed a
#     vector of size minibatch x (params['source-window']+params['target-window'])
with tf.device('/gpu:0' if params['use-gpu'] else '/cpu:0'):
    input_indices = tf.placeholder(tf.int64, shape=[None, params['source-window']+params['target-window']])

#The output will be given a minibatch vector of correct indicies
with tf.device('/gpu:0' if params['use-gpu'] else '/cpu:0'):
    correct_output = tf.placeholder(tf.int64,shape=[None])

#For passing in the learning rate and dropout rate
with tf.device('/gpu:0' if params['use-gpu'] else '/cpu:0'):
    learning_rate = tf.placeholder(params['datatype'],shape=[])
    dropout_rate = tf.placeholder(params['datatype'],shape=[])


# In[10]:

def param_init(params,shape=None,name=None,datatype=None):
    if datatype == None:
        datatype = params['datatype']
    assert shape != None,"Error shape cannot be None in param_init"
    if params['init-method'] == 'uniform':
        return tf.random_uniform(shape, minval=-1*params['init-range'],maxval=params['init-range'],                    dtype=datatype, seed=params['seed'], name=name)
    else:
        print "ERROR this init-method has not been created yet"


# In[11]:

#Parameters for the source and target embeddings
with tf.device('/gpu:0' if params['use-gpu'] else '/cpu:0'):
    source_emb_matrix = tf.Variable(param_init(params,shape=[params['source-vocab'],params['input-emb-size']],name='src-emb'))
    target_emb_matrix = tf.Variable(param_init(params,shape=[params['target-vocab'],params['input-emb-size']],name='tgt-emb'))


# In[12]:

#Do the one-hot emebedding lookups
src_embed = tf.nn.embedding_lookup(source_emb_matrix, tf.slice(input_indices,[0,0],[-1,params['source-window']]))
tgt_embed = tf.nn.embedding_lookup(target_emb_matrix, tf.slice(input_indices,[0,params['source-window']],                                                                [-1,params['target-window']]))


# In[13]:

#Now reshape to be able to feed through non-linearity
concat_embed = tf.concat(1, [src_embed, tgt_embed])
concat_embed = tf.reshape(concat_embed,[-1,params['input-emb-size']*(params['source-window']+params['target-window'])])
concat_embed = tf.nn.dropout(concat_embed, 1-dropout_rate) #Pass in the keep prob


# In[14]:

#First Layer
with tf.device('/gpu:0' if params['use-gpu'] else '/cpu:0'):
    layer_1_weights = tf.Variable(param_init(params,        shape=[params['input-emb-size']*(params['source-window']+params['target-window']),params['hiddenstate-size']],name='lyr-1'))
    layer_1_bias = tf.Variable(param_init(params,shape=[params['hiddenstate-size']],name='lyr-1-bias'))
layer_1_output = tf.nn.relu(tf.matmul(concat_embed,layer_1_weights)+layer_1_bias)
layer_1_output = tf.nn.dropout(layer_1_output, 1-dropout_rate) #Pass in the keep prob


# In[15]:

#Second layer
with tf.device('/gpu:0' if params['use-gpu'] else '/cpu:0'):
    layer_2_weights = tf.Variable(param_init(params,shape=[params['hiddenstate-size'],params['output-emb-size']],                            name='lyr-2'))
    layer_2_bias = tf.Variable(param_init(params,shape=[params['output-emb-size']],name='lyr-2-bias'))
layer_2_output = tf.nn.relu(tf.matmul(layer_1_output,layer_2_weights)+layer_2_bias)
layer_2_output = tf.nn.dropout(layer_2_output, 1-dropout_rate) #Pass in the keep prob


# In[16]:

#Softmax layer
with tf.device('/gpu:0' if params['use-gpu'] else '/cpu:0'):
    softmax_weights = tf.Variable(param_init(params,shape=[params['output-emb-size'],params['target-vocab']],                                  name='softmax-weights')) 
    softmax_bias = tf.Variable(param_init(params,shape=[params['target-vocab']],name='softmax-bias'))
unscaled_final_output = tf.matmul(layer_2_output, softmax_weights) + softmax_bias
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_final_output,correct_output))


# In[17]:

#Optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# In[18]:

#Initiailize all the variables
sess.run(tf.initialize_all_variables())


# In[ ]:

#Additional variables for tracking loss
val_perplexities = [] #stores the validation perplexities each validation point


# In[ ]:

print "Training for",params['epochs'],"epochs (",params['epochs']*params['minibatches-per-epoch'],"minibatches)"
print "All current parameters:"
print params,'\n\n'
print "-"*10,'beginning training','-'*10
start_train = time.time()
data_factory.prep_train() #prep the data loader for training, also stores timing info
for i in range(params['epochs']*params['minibatches-per-epoch']):
    #get the minibatch
    curr_minibatch,eval_val = data_factory.get_minibatch()
    if eval_val:
        start_val = time.time()
        log_sum = 0
        total_words = 0
        for val_batch in data_factory.get_val_data_gen():
            #print val_batch[:,:params['source-window']+params['target-window']].shape
            #print np.squeeze(val_batch[:,params['source-window']+params['target-window']:]).shape
            input_val_batch = val_batch[:,:params['source-window']+params['target-window']]
            output_val_batch = np.squeeze(np.copy(val_batch[:,params['source-window']+params['target-window']:]))
            log_sum+=loss.eval(feed_dict={input_indices:input_val_batch,
                        correct_output:output_val_batch, 
                        dropout_rate:0.0})
            total_words+=val_batch.shape[0]
        log_sum = (log_sum/np.log(2.0))/total_words
        print "Perplexity on validation set:",2**log_sum
        val_perplexities.append(2**log_sum)
        if (len(val_perplexities) > 1) and (val_perplexities[-2] + params['epsilon-criteria'] < val_perplexities[-1]):
            params['learning-rate']*=params['decrease-factor']
            print 'Decreased learning rate to:',params['learning-rate']
        end_val = time.time()
        print "Time for perplexity on dev set (minutes):",(end_val - start_val)/60.0
    #Now update the gradients for this training batch
    input_train_batch = curr_minibatch[:,:params['source-window']+params['target-window']]
    output_train_batch = np.squeeze(np.copy(curr_minibatch[:,params['source-window']+params['target-window']:]))
    train_op.run(feed_dict={input_indices:input_train_batch,
                correct_output:output_train_batch,\
                dropout_rate:params['dropout-rate'],learning_rate:params['learning-rate'] })

end_train = time.time()
print "Time for total training (minutes):",(end_train - start_train)/60.0


# In[ ]:



