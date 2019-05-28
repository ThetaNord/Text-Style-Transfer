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
flags.DEFINE_string('output', None, 'Output file name')
#flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

# Class for training word2vec embeddings
class EmbeddingModel:

	def __init__(self, model_name, vocab, n):
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
	
	def __init__(self, input_name, output_name, n):
		self.input_model_name = input_name
		self.output_model_name = output_name
		self.dimensions = n

	def init_model(self):
		self.model = Sequential()
		#self.model.add(Dense(50, input_shape=(self.dimensions,)))
		#self.model.add(Dense(self.dimensions))
		self.model.add(Dense(self.dimensions, input_shape=(self.dimensions,)))
		self.model.compile(optimizer='adam', loss='cosine_proximity', metrics=['cosine_proximity'])
		print("Initialized transform model")
		
	def train(self, X_train, y_train, epochs, save_model=True):
		self.model.fit(X_train, y_train, epochs=epochs)
		if save_model:
			self.save_model()
			
	def transform(self, input):
		transform = self.model.predict(input.reshape((1, -1)))
		return transform
		
	def get_model_path(self):
		cwd = os.getcwd()
		model_path = os.path.join(cwd, 'models', self.input_model_name, 'to_' + self.output_model_name + '.h5')
		return model_path
		
	def load_model(self):
		self.model = load_model(self.get_model_path())
		print("Loaded existing transform model")
	
	def save_model(self):
		self.model.save(self.get_model_path())
		print("Saved transform model")
	
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
	model = EmbeddingModel(model_name, vocab, FLAGS.dimensions)
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
	
def load_vocabulary_embedding(model_name):
	# Load embedding file
	cwd = os.getcwd()
	embed_path = os.path.join(cwd, 'models', model_name, 'embeddings.npy')
	embeddings = np.load(embed_path)
	return embeddings
	
def get_vocabulary_embedding(model_name):
	try:
		embedding = load_vocabulary_embedding(model_name)
	except FileNotFoundError:
		embedding = create_vocabulary_embedding(model_name)
	return embedding
	
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
		
def find_nearest_word(vocab, embeddings, input_coordinates):
	word = None
	score = 0
	for i in range(vocab.count):
		new_score = cosine_similarity(embeddings[i], input_coordinates)[0][0]
		if new_score > score:
			word = vocab.vocabulary[i]
			score = new_score
	return word, score
	
def train_transform_model(input_model_name, output_model_name):
	# Load vocabularies for both models
	input_vocabulary = get_vocabulary(input_model_name)
	output_vocabulary = get_vocabulary(output_model_name)
	# Load embeddings for both models
	input_embeddings = get_vocabulary_embedding(input_model_name)
	output_embeddings = get_vocabulary_embedding(output_model_name)
	# Create transform model
	transform_model = TransformModel(input_model_name, output_model_name, FLAGS.dimensions)
	transform_model.init_model()
	# Identify words shared between vocabularies and create training data
	X_train = []
	y_train = []
	for i in range(input_vocabulary.count):
		w = input_vocabulary.vocabulary[i]
		other_idx = output_vocabulary.get_index(w)
		if other_idx:
			X_train.append(input_embeddings[i])
			y_train.append(output_embeddings[other_idx])
	X_train = np.squeeze(np.asarray(X_train))
	y_train = np.squeeze(np.asarray(y_train))
	# Train the model
	transform_model.train(X_train, y_train, FLAGS.epochs, save_model=True)
	return transform_model

def get_transfrom_model(input_model_name, output_model_name):
	transform_model = TransformModel(input_model_name, output_model_name,FLAGS.dimensions)
	try:
		transform_model.load_model()
	except OSError:
		transform_model = train_transform_model(input_model_name, output_model_name)
	return transform_model
	
def load_input_text(input_file):
	raw = open(input_file).read()
	tokens = word_tokenize(raw)
	cleaned_tokens = [w.lower() for w in tokens]
	text = nltk.Text(cleaned_tokens)
	return text
	
def transfer_style(input_file, input_model_name, output_model_name, output_file=None):
	# Get vocabularies for both models
	input_vocabulary = get_vocabulary(input_model_name)
	output_vocabulary = get_vocabulary(output_model_name)
	# Get embeddings for both models
	input_embeddings = get_vocabulary_embedding(input_model_name)
	output_embeddings = get_vocabulary_embedding(output_model_name)
	# Get the transform model from input to output
	transform_model = get_transfrom_model(input_model_name, output_model_name)
	# Load input text
	text = load_input_text(input_file)
	new_tokens = [None] * len(text)
	# For each word in input text
	for i in range(len(text)):
		word = text[i]
		# Find if it exists in input corpus
		if word in input_vocabulary.vocabulary:
			print('"'+word+'" replaced with "', end="")
			# Transform to output embedding
			idx = input_vocabulary.get_index(word)
			embedding = input_embeddings[idx]
			transformed_embedding = transform_model.transform(embedding)
			# Find nearest neighbour in output space
			word, score = find_nearest_word(output_vocabulary, output_embeddings, transformed_embedding)
			print(word+'" ('+str(score)+')')
		# Replace the word
		new_tokens[i] = word
	new_text = nltk.Text(new_tokens)
	if not output_file and not output_file == "":
		# Print new text
		for word in new_text:
			print(word, end=' ')
	else:
		text_file = open(output_file, "w")
		for word in new_text:
			text_file.write(word + " ")
		text_file.close()
	
def main(argv):
	# Check that relevant parameters have been given
	if (len(sys.argv) < 2):
		print("Usage: python text-style-transfer.py <operation> [other arguments]")
		sys.exit()
	# Interpret command line arguments and call correct function
	operation = sys.argv[1]
	print("Dimensions: " + str(FLAGS.dimensions))
	if operation == "train-transform":
		input_model_name = sys.argv[2]
		output_model_name = sys.argv[3]
		train_transform_model(input_model_name, output_model_name)
	elif operation == "style-transfer":
		input_file = sys.argv[2]
		input_model_name = sys.argv[3]
		output_model_name = sys.argv[4]
		transfer_style(input_file, input_model_name, output_model_name, FLAGS.output)
	elif operation == "proximity-test":
		model_name = sys.argv[2]
		embeddings = get_vocabulary_embedding(model_name)
		find_closest_words(model_name, embeddings, "jesus")
		find_closest_words(model_name, embeddings, "god")
		find_closest_words(model_name, embeddings, "it")
		find_closest_words(model_name, embeddings, "was")
	
if __name__== "__main__":
	app.run(main)
