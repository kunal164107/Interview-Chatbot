import warnings
warnings.filterwarnings("ignore")
import copy
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.layers import DenseLayer, EmbeddingInputlayer, Seq2Seq, retrieve_seq_length_op2
from data.twitter import data


class chatbot(object):
	word2idx = None
	idx2word = None
	unk_id = None
	pad_id = None
	start_id = None
	end_id = None
	sess = None
	encode_seqs2 = None
	decode_seqs2 = None
	net = None
	net_rnn = None
	y = None
	data_corpus = "twitter"
	sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	
	def __init__(self):

		metadata, trainX, trainY, testX, testY, validX, validY = initial_setup(self.data_corpus)

		src_vocab_size = len(metadata['idx2w']) # 1686 (0~1685)
		emb_dim = 1024

		self.word2idx = metadata['w2idx']   # dict  word 2 index
		self.idx2word = metadata['idx2w']   # list index 2 word

		self.unk_id = self.word2idx['unk']   # 1
		self.pad_id = self.word2idx['_']     # 0

		self.start_id = src_vocab_size  # 1686
		self.end_id = src_vocab_size + 1  # 1687

		self.word2idx.update({'start_id': self.start_id})
		self.word2idx.update({'end_id': self.end_id})
		self.idx2word = self.idx2word + ['start_id', 'end_id']

		src_vocab_size = tgt_vocab_size = src_vocab_size + 2

		# Init Session
		tf.reset_default_graph()
		self.sess = tf.Session(config=self.sess_config)

		# testing Data Placeholders
		self.encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
		self.decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")

		self.net, self.net_rnn = create_model(self.encode_seqs2, self.decode_seqs2, src_vocab_size, emb_dim, is_train=False, reuse=True)
		self.y = tf.nn.softmax(self.net.outputs)

		# Init Vars
		self.sess.run(tf.global_variables_initializer())

		# Load Model
		tl.files.load_and_assign_npz(sess=self.sess, name='model.npz', network=self.net)
		
	def testing(self,seed):
		seed_id = [self.word2idx.get(w, self.unk_id) for w in seed.split(" ")]

		# Encode and get state
		state = self.sess.run(self.net_rnn.final_state_encode,
						{self.encode_seqs2: [seed_id]})
		# Decode, feed start_id and get first word [https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py]
		o, state = self.sess.run([self.y, self.net_rnn.final_state_decode],
						{self.net_rnn.initial_state_decode: state,
						self.decode_seqs2: [[self.start_id]]})
		w_id = tl.nlp.sample_top(o[0], top_k=3)
		w = self.idx2word[w_id]
		# Decode and feed state iteratively
		sentence = [w]
		for _ in range(30): # max sentence length
			o, state = self.sess.run([self.y, self.net_rnn.final_state_decode],{self.net_rnn.initial_state_decode: state,self.decode_seqs2: [[w_id]]})
			w_id = tl.nlp.sample_top(o[0], top_k=2)
			w = self.idx2word[w_id]
			if w_id == self.end_id:
				break
			sentence = sentence + [w]
		return sentence

	def test_output(self,ques):
		pickle_in = open("unique_words.pickle","rb")
		vocablist = pickle.load(pickle_in)
		
		answer = ''
		input_seq_list = ques.split(' ')
		
		count = 0
		for word in input_seq_list:
			if word in vocablist:
				# print("word " + word)
				count = count + 1
		
		# print("count "+str(count))
		# print("len(input_seq_list) "+str(len(input_seq_list)))
		# print("input_seq_list "+ input_seq_list[0])
		
		if len(input_seq_list)==1 and input_seq_list[0]=='':
			answer = 'Please type Something'
		elif count==0:
			answer = 'Out of Context Question'
		else:
			answer = self.test_run(ques)
		return answer
	
	def test_run(self,ques):
		p=self.testing(ques)
		answer = ' '.join(p)
		return answer

	def __del__(self):
		self.sess.close()


"""
Creates the LSTM Model
"""
def create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, is_train=True, reuse=False):
	with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
		with tf.variable_scope("embedding") as vs:
			net_encode = EmbeddingInputlayer(
				inputs = encode_seqs,
				vocabulary_size = src_vocab_size,
				embedding_size = emb_dim,
				name = 'seq_embedding')
			vs.reuse_variables()
			net_decode = EmbeddingInputlayer(
				inputs = decode_seqs,	
				vocabulary_size = src_vocab_size,
				embedding_size = emb_dim,
				name = 'seq_embedding')
			
		net_rnn = Seq2Seq(net_encode, net_decode,
				cell_fn = tf.nn.rnn_cell.LSTMCell,
				n_hidden = emb_dim,
				initializer = tf.random_uniform_initializer(-0.1, 0.1),
				encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
				decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
				initial_state_encode = None,
				dropout = (0.5 if is_train else None),
				n_layer = 3,
				return_seq_2d = True,
				name = 'seq2seq')

		net_out = DenseLayer(net_rnn, n_units=src_vocab_size, act=tf.identity, name='output')
	return net_out, net_rnn

"""
Initial Setup
"""
def initial_setup(data_corpus):
	metadata, idx_q, idx_a = data.load_data(PATH='data/{}/'.format(data_corpus))
	(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

	# Remove padding from all (tl.prepro.remove_pad_sequences())
	trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
	trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
	testX = tl.prepro.remove_pad_sequences(testX.tolist())
	testY = tl.prepro.remove_pad_sequences(testY.tolist())
	validX = tl.prepro.remove_pad_sequences(validX.tolist())
	validY = tl.prepro.remove_pad_sequences(validY.tolist())
	return metadata, trainX, trainY, testX, testY, validX, validY


def test_model():
	try:
		
		ob = chatbot()
		input_seq = input("Query: ")
		answer = ob.test_output(input_seq)
		print(answer)
	except KeyboardInterrupt:
		print('Aborted!')

if __name__ == '__main__':
	test_model()