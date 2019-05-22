import sys, os
#import unidecode

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation

import nltk
#nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, 'The name of the model')
flags.DEFINE_integer('window_size', 2, 'Size of context window, defaults to 2', lower_bound=1)
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train, defaults to 10', lower_bound=1)
flags.DEFINE_integer('dimensions', 100, 'Number of dimensions to embed words in, defaults to 100', lower_bound=1)
#flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

# Class for training word2vec embeddings
class EmbeddingModel:

	def __init__(self, model_name, vocab, n=100):
		self.vocabulary = vocab
		self.model_name = model_name
		self.dimensions = n

	def init_model(self):
		self.model = Sequential()
		self.model.add(Dense(self.dimensions, input_shape=(self.vocabulary.count,), name="embedding"))
		self.model.add(Dense(self.vocabulary.count))
		self.model.add(Activation("softmax"))
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
		self.embedding_model = Model(inputs=self.model.input, outputs=self.model.get_layer("embedding").output)
		print("Initialized embedding model")
		
	def train(self, X_train, y_train, epochs, save_model=True):
		self.model.fit(X_train, y_train, epochs=epochs)
		if save_model:
			self.save_model()
		
	def get_embedding(self, word):
		embedding = None
		if self.vocabulary.contains(word):
			input = self.vocabulary.word2onehot(word)
			embedding = self.embedding_model.predict(input.reshape((1, -1)))
		return embedding
		
	def get_model_path(self):
		cwd = os.getcwd()
		model_path = os.path.join(cwd, 'models', self.model_name, 'embedding_model.h5')
		return model_path
		
	def load_model(self):
		self.model = load_model(self.get_model_path())
		print(self.model.summary())
		self.embedding_model = Model(inputs=self.model.input, outputs=self.model.get_layer("embedding").output)
		print("Loaded existing embedding model")
	
	def save_model(self):
		self.model.save(self.get_model_path())
		print("Saved embedding model")
		
		
# Model for transforms between to word2vec embeddings (order matters)
class TransformModel:
	
	def __init__(self):
		self.input_name = "name"
		self.output_name = "name"
	
class Vocabulary:
	
	def __init__(self, vocab):
		self.vocabulary = vocab
		self.count = len(vocab)
		
	def contains(self, word):
		word = word.lower()
		return word in self.vocabulary
		
	def get_index(self, word):
		word = word.lower()
		if word in self.vocabulary:
			return self.vocabulary.index(word)
		
	def word2onehot(self, word):
		onehot = np.zeros(self.count)
		try:
			word_index = self.vocabulary.index(word.lower())
		except ValueError:
			print('Word "', word.lower(), '" not found in vocabulary! Vocabulary might be outdated or corrupted.')
			print("Try deleting the vocabulary file and running the program again.")
			sys.exit()
		onehot[word_index] = 1;
		return onehot
		
	def context2onehot(self, context):
		onehot = np.zeros(self.count)
		for word in context:
			try:
				word_index = self.vocabulary.index(word.lower())
			except ValueError:
				print('Word "', word.lower(), '" not found in vocabulary! Vocabulary might be outdated or corrupted.')
				print("Try deleting the vocabulary file and running the program again.")
				sys.exit()
			onehot[word_index] = 1;
		return onehot

def load_corpus(model_name):
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
	#raw = unidecode.unidecode(raw)
	tokens = word_tokenize(raw)
	filtered_tokens = [t for t in tokens if t not in "''``.,!?;:--()"] #'".,!?;:-—'
	text = nltk.Text(filtered_tokens)
	return text

def get_vocabulary(model_name):
	try:
		vocab = load_vocabulary(model_name)
	except FileNotFoundError:
		vocab = create_vocabulary(model_name)
	return vocab
	
def create_vocabulary(model_name):
	# Load text from corpus
	text = load_corpus(model_name)
	# Create vocabulary
	words = [w.lower() for w in text]
	vocab = Vocabulary(sorted(set(words)))
	#print(len(vocab.vocabulary))
	#print(vocab.vocabulary[:100])
	print("Created vocabulary")
	# Save vocabulary to file
	cwd = os.getcwd()
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
	#print(vocab.vocabulary[:50])
	return vocab
	
def create_training_data(model_name, vocab, window_size):
	text = load_corpus(model_name)
	X_train = []
	y_train = []
	for i in range(len(text)):
		context = []
		for j in range(i-window_size, i+window_size+1):
			if j != i and j >= 0 and j < len(text):
				context.append(text[j])
		X_train.append(vocab.word2onehot(text[i]))
		y_train.append(vocab.context2onehot(context))
	print("Created training samples")
	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	print("Training data created")
	return X_train, y_train
	
def train_embedding_model(model_name, window_size=2, epochs=10, vocab=None):
	# Get vocabulary
	if not vocab:
		vocab = get_vocabulary(model_name)
	# Create skip-grams, i.e. the training data
	X_train, y_train = create_training_data(model_name, vocab, window_size)
	#print(X_train.shape)
	#print("Samples:", len(training_data))
	# Create model
	model = EmbeddingModel(model_name, vocab)
	model.init_model()
	# Train model
	print("Starting training")
	model.train(X_train, y_train, epochs=epochs)
	return model
	
def create_vocabulary_embedding(model_name):
	# Load or create vocabulary
	vocab = get_vocabulary(model_name)
	# Load or create embedding model
	embedding_model = EmbeddingModel(model_name, vocab, FLAGS.dimensions)
	try:
		embedding_model.load_model()
	except OSError:
		print("Could not load embedding model. Attempting training.")
		embedding_model = train_embedding_model(model_name, FLAGS.window_size, FLAGS.epochs, vocab)
	# Create a list of embeddings for each word in the vocabulary
	embeddings = []
	for word in vocab.vocabulary:
		embeddings.append(embedding_model.get_embedding(word))
	embeddings = np.array(embeddings)
	# Save embedding to file
	cwd = os.getcwd()
	embed_path = os.path.join(cwd, 'models', model_name, 'embeddings.npy')
	np.save(embed_path, embeddings)
	print("Embeddings saved")
	return embeddings

def find_closest_word(model_name, embeddings, word, n=5):
	vocab = get_vocabulary(model_name)
	index = vocab.get_index(word)
	score = 0
	output = ""
	for w in vocab.vocabulary:
		new_index = vocab.get_index(w)
		if new_index != index:
			new_score = cosine_similarity(embeddings[index], embeddings[new_index])
			if new_score > score:
				score = new_score
				output = w
	print("Closest match is", output, "with a similarity of", score)
	
def find_closest_words(model_name, embeddings, word, n=5):
	vocab = get_vocabulary(model_name)
	index = vocab.get_index(word)
	scores = []
	for i in range(vocab.count):
		scores.append(cosine_similarity(embeddings[i], embeddings[index])[0][0])
	scores = np.array(scores)
	print(scores.shape)
	ind = np.argpartition(scores, -(n+1))[-(n+1):]
	ind = ind[np.argsort(scores[ind])[::-1]]
	print("Closest matches for '", word, "' are:")
	for i in ind[1:]:
		print(vocab.vocabulary[i], ", score:", scores[i])
	
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
	
def main(argv):
	# Check that relevant parameters have been given
	if (len(sys.argv) < 2):
		print("Usage: python text-style-transfer.py <model_name>")
		sys.exit()
	# Interpret command line argument
	model_name = sys.argv[1]
	# Call correct function
	embeddings = create_vocabulary_embedding(model_name)
	find_closest_words(model_name, embeddings, "jesus")
	
if __name__== "__main__":
	app.run(main)
