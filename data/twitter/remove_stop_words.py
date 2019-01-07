from __future__ import absolute_import
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'final1.txt'

limit = {
		'maxq' : 20,
		'minq' : 0,
		'maxa' : 20,
		'mina' : 3
		}

UNK = 'unk'
VOCAB_SIZE = 6000

import random
import sys
import nltk
import itertools
from collections import defaultdict
import numpy as np
import pickle

def ddefault():
    return 1

'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
	return open(filename,encoding='utf8').read().split('\n')[:-1]    # require in pytnon 3 ,encoding='utf8'


'''
 split sentences in one line
  into multiple lines
    return [list of lines]

'''

def split_line(line):
	return line.split('.')

def remove_stopwords(line):
	stop_words = set(stopwords.words('english')) 
	word_tokens = word_tokenize(line) 
	  
	filtered_sentence = [w for w in word_tokens if not w in stop_words] 
	filtered_sentence = []   
	for w in word_tokens: 
		if w not in stop_words: 
			filtered_sentence.append(w) 
	
	return ' '.join(filtered_sentence)
 
         
'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, whitelist):
	return ''.join([ ch for ch in line if ch in whitelist ])

	

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def vocablist(tokenized_sentences, vocab_size):
	# get frequency distribution
	freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	vocab = freq_dist.most_common(vocab_size)
	vocablist = []
	for i in range(0,len(vocab)):
		vocablist.append(vocab[i][0]) 
	
	assert len(vocab) == len(vocablist)

	pickle_out = open("unique_words.pickle","wb")
	pickle.dump(vocablist, pickle_out)
	pickle_out.close()


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(sequences):
	filtered_q, filtered_a = [], []
	raw_data_len = len(sequences)//2

	for i in range(0, len(sequences), 2):
		qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
		if qlen >= limit['minq'] and qlen <= limit['maxq']:
			if alen >= limit['mina'] and alen <= limit['maxa']:
				filtered_q.append(sequences[i])
				filtered_a.append(sequences[i+1])
		
	# print the fraction of the original data, filtered
	filt_data_len = len(filtered_q)
	print(str(len(filtered_q)))
	filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
	print(str(filtered) + '% filtered from original data')

	return filtered_q, filtered_a


def process_data():

	print('\n>> Read lines from file')
	lines = read_lines(filename=FILENAME)

	# change to lower case (just for en)
	lines = [ line.lower() for line in lines ]

	print('\n:: Sample from read(p) lines')
	print(lines[121:125])

	# filter out unnecessary characters
	print('\n>> Filter lines')
	lines = [ filter_line(line, EN_WHITELIST) for line in lines ]
	print(lines[121:125])

	print('\n>> Remove Stop words to get vocab list')
	lines = [ remove_stopwords(line) for line in lines ]
	print(lines[121:125])

	# filter out too long or too short sequences
	print('\n>> 2nd layer of filtering')
	qlines, alines = filter_data(lines)
	print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
	print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))


	# convert list of [lines of text] into list of [list of words ]
	print('\n>> Segment lines into words')
	qtokenized = [ wordlist.split(' ') for wordlist in qlines ]
	atokenized = [ wordlist.split(' ') for wordlist in alines ]
	print('\n:: Sample from segmented list of words')
	print('\nq : {0} ; a : {1}'.format(qtokenized[60], atokenized[60]))
	print('\nq : {0} ; a : {1}'.format(qtokenized[61], atokenized[61]))


	# indexing -> idx2w, w2idx : en/ta
	print('\n >> Index words')
	vocablist( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

if __name__ == '__main__':
	process_data()
