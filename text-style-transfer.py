import sys, os

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

import nltk
#nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords

# Class for training word2vec embeddings
class EmbeddingModel:

	def __init__(self, vocab, n=100):
		self.model = self.create_model(vocab, n)
		self.dimensions = n

	def create_model(self, vocab, n):
		model = Sequential()
		model.add(Dense(n, input_shape=(vocab.count,), name="embedding"))
		model.add(Dense(vocab.count))
		model.add(Activation("softmax"))
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
		return model
		
# Model for transforms between to word2vec embeddings (order matters)
class TransformModel:
	
	def __init__(self):
		self.input_name = "name"
		self.output_name = "name"
	
class Vocabulary:
	
	def __init__(self, vocab):
		self.vocabulary = vocab
		self.count = len(vocab)
		
	def word2onehot(self, word):
		onehot = None
		#if self.vocabulary.contains(word.lower()):
		onehot = np.zeros(len(self.vocabulary))
		word_index = self.vocabulary.index(word.lower())
		onehot[word_index] = 1;
		return onehot
		
	def context2onehot(self, context):
		onehot = np.zeros(len(self.vocabulary))
		for word in context:
			#if self.vocabulary.contains(word.lower()):
			word_index = self.vocabulary.index(word.lower())
			onehot[word_index] = 1;
		return onehot

def create_vocabulary(model_name):
	# Load source file
	cwd = os.getcwd()
	corpus_path = os.path.join(cwd, 'models', model_name, 'corpus.txt')
	raw = None
	try:
		raw = open(corpus_path).read()
	except FileNotFoundError:
		print("Could not open corpus file in location", corpus_path)
		print("Make sure that the file exists and is accessible")
		sys.exit()
	# Preprocess source file
	tokens = word_tokenize(raw)
	filtered_tokens = [t for t in tokens if t not in "''``.,!?;:--()"] #'".,!?;:-—'
	text = nltk.Text(filtered_tokens)
	# Create vocabulary
	words = [w.lower() for w in text]
	vocab = Vocabulary(sorted(set(words)))
	#print(len(vocab.vocabulary))
	#print(vocab.vocabulary[:100])
	print("Created vocabulary")
	# Save vocabulary to file
	vocab_path = os.path.join(cwd, 'models', model_name, 'vocabulary.txt')
	vocab_file = open(vocab_path, "w")
	for word in vocab.vocabulary:
		vocab_file.write(word + "\n")
	print("Saved vocabulary")
	return vocab
	
def load_vocabulary(model_name):
	# Load vocabulary file
	cwd = os.getcwd()
	vocab_path = os.path.join(cwd, 'models', model_name, 'vocabulary.txt')
	vocab_file = open(vocab_path)
	words = []
	for line in vocab_file.readlines():
		words.append(line.rstrip())
	vocab = Vocabulary(words)
	print("Loaded vocabulary")
	print(vocab.vocabulary[:50])
	return vocab
		
def train_embedding(model_name, window_size=2, epochs=10):
	# Create vocabulary
	vocab = create_vocabulary(model_name)
	#print(len(vocab.vocabulary))
	#print(vocab.vocabulary[:100])
	print("Created vocabulary")
	# Create skip-grams, i.e. the training data
	X_train = np.empty()
	y_train = np.empty()
	for i in range(len(text)):
		context = []
		for j in range(i-window_size, i+window_size+1):
			if j != i and j >= 0 and j < len(text):
				context.append(text[j])
		X_train.append(vocab.word2onehot(text[i]))
		y_train.append(vocab.context2onehot(context))
	print("Created training samples")
	#print("Samples:", len(training_data))
	# Create model
	model = EmbeddingModel(vocab)
	# Train model
	print("Starting training")
	model.model.fit(X_train, y_train, epochs=epochs)
	# Save model to file
	# Save vocabulary to file
	# Save embedding to file
	return

def train_transform(input_model_name, output_model_name, epochs=50):
	# Load embeddings from file
	# Create transform model
	# Identify words shared between vocabularies and create training data
	# Train the model
	# Store the model to file
	return

def transfer_style(input_file, input_model_name, output_model_name, output_file=None):
	# Load embeddings for both models
	# Load the transform model from input to output
	# For each word in input text
		# Find if it exists in input corpus
		# Transform to output embedding
		# Find nearest neighbour in output space
		# Replace the word
	# Print new text
	return
	
def main():
	# Check that relevant parameters have been given
	if (len(sys.argv) < 2):
		print("Usage: python text-style-transfer.py <model_name>")
		sys.exit()
	# Interpret command line argument
	model_name = sys.argv[1]
	# Call correct function
	#train_embedding(model_name)
	load_vocabulary(model_name)
	
if __name__== "__main__":
	main()
