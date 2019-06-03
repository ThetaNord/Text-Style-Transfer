import sys, os, io
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
flags.DEFINE_integer('window_size', 10, 'Size of context window, defaults to 10', lower_bound=1)
flags.DEFINE_integer('batch_size', 32, 'Batch size for embedding training, defaults to 32', lower_bound=1)
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train, defaults to 20', lower_bound=1)
flags.DEFINE_integer('dimensions', 250, 'Number of dimensions to embed words in, defaults to 250', lower_bound=1)
flags.DEFINE_string('output', None, 'Output file name')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

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
			
	def train_generator(self, generator_func, step_count, epochs, save_model=True):
		self.model.fit_generator(generator=generator_func, steps_per_epoch=step_count, epochs=epochs)
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
		model_path = os.path.join(cwd, 'models', self.model_name, 'embedding_model_' + str(FLAGS.dimensions) + '.h5')
		return model_path
		
	def load_model(self):
		self.model = load_model(self.get_model_path())
		if FLAGS.debug: print(self.model.summary())
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
		model_path = os.path.join(cwd, 'models', self.input_model_name, 'to_' + self.output_model_name + '_' + str(FLAGS.dimensions) + '.h5')
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
		f = io.open(corpus_path, mode="r", encoding="utf-8")
		raw = f.read()
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
	vocab_file = open(vocab_path, "wb")
	for word in vocab.vocabulary:
		vocab_file.write((word + "\n").encode('UTF-8'))
	print("Saved vocabulary")
	return vocab
	
def load_vocabulary(model_name):
	# Load vocabulary file
	cwd = os.getcwd()
	vocab_path = os.path.join(cwd, 'models', model_name, 'vocabulary.txt')
	vocab_file = io.open(vocab_path, mode="r", encoding="utf-8")
	words = []
	for line in vocab_file.readlines():
		words.append(line.rstrip())
	vocab = Vocabulary(words)
	print("Loaded vocabulary")
	#print(vocab.vocabulary[:50])
	return vocab

def generate_data(model_name, window_size, batch_size):
	text = load_corpus(model_name)
	vocab = load_vocabulary(model_name)
	ids = np.array(range(len(text)))
	np.random.shuffle(ids)
	i = 0
	while True:
		X = []
		y = []
		for b in range(batch_size):
			if i == len(ids):
				i = 0
				np.random.shuffle(ids)
			idx = ids[i]
			i += 1
			X.append(vocab.word2onehot(text[idx]))
			context = []
			for j in range(idx-window_size, idx+window_size+1):
				if j != idx and j >= 0 and j < len(text):
					context.append(text[j])
			y.append(vocab.context2onehot(context))
		yield (np.array(X), np.array(y))
	
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
	
def train_embedding_model(model_name, window_size, epochs, batch_size, vocab=None):
	# Get vocabulary
	if not vocab:
		vocab = get_vocabulary(model_name)
	# Create skip-grams, i.e. the training data
	# X_train, y_train = create_training_data(model_name, vocab, window_size)
	#print(X_train.shape)
	#print("Samples:", len(training_data))
	# Create model
	model = EmbeddingModel(model_name, vocab, FLAGS.dimensions)
	model.init_model()
	# Train model
	print("Starting training")
	#model.train(X_train, y_train, epochs=epochs)
	text_length = len(load_corpus(model_name))
	step_count = text_length//batch_size
	print("T,B,S:" + str(text_length) + ", " + str(batch_size) + ", " + str(step_count))
	model.train_generator(generate_data(model_name, window_size, batch_size), step_count, epochs)
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
		embedding_model = train_embedding_model(model_name, FLAGS.window_size, FLAGS.epochs, FLAGS.batch_size, vocab)
	# Create a list of embeddings for each word in the vocabulary
	embeddings = []
	for word in vocab.vocabulary:
		embeddings.append(embedding_model.get_embedding(word))
	embeddings = np.array(embeddings)
	# Save embedding to file
	cwd = os.getcwd()
	embed_path = os.path.join(cwd, 'models', model_name, 'embeddings_'+ str(FLAGS.dimensions) +'.npy')
	np.save(embed_path, embeddings)
	print("Embeddings saved")
	return embeddings
	
def load_vocabulary_embedding(model_name):
	# Load embedding file
	cwd = os.getcwd()
	embed_path = os.path.join(cwd, 'models', model_name, 'embeddings_' + str(FLAGS.dimensions) +'.npy')
	embeddings = np.load(embed_path)
	return embeddings
	
def get_vocabulary_embedding(model_name):
	try:
		embedding = load_vocabulary_embedding(model_name)
	except FileNotFoundError:
		embedding = create_vocabulary_embedding(model_name)
	return embedding
	
def proximity_test(model_name, n=5):
	vocab = get_vocabulary(model_name)
	embeddings = get_vocabulary_embedding(model_name)
	for word in vocab.vocabulary:
		find_closest_words(word, vocab, embeddings, n)

def find_closest_words(word, vocab, embeddings, n=5):
	index = vocab.get_index(word)
	scores = []
	for i in range(vocab.count):
		scores.append(cosine_similarity(embeddings[i], embeddings[index])[0][0])
	scores = np.array(scores)
	if FLAGS.debug: print(scores.shape)
	ind = np.argpartition(scores, -(n+1))[-(n+1):]
	ind = ind[np.argsort(scores[ind])[::-1]]
	print(word + " - Closest matches are:")
	for i in ind[1:]:
		print("\t" + vocab.vocabulary[i], ", score:", scores[i])
		
def find_nearest_word(vocab, embeddings, input_coordinates):
	word = None
	score = 0
	for i in range(vocab.count):
		new_score = cosine_similarity(embeddings[i], input_coordinates)[0][0]
		if new_score > score:
			word = vocab.vocabulary[i]
			score = new_score
	return word, score

def get_transform_data(input_model_name, output_model_name, threshold_count=10):
	# Load output corpus
	text = load_corpus(output_model_name)
	words = [w.lower() for w in text]
	# Load vocabularies for both models
	input_vocabulary = get_vocabulary(input_model_name)
	output_vocabulary = get_vocabulary(output_model_name)
	# Load embeddings for both models
	input_embeddings = get_vocabulary_embedding(input_model_name)
	output_embeddings = get_vocabulary_embedding(output_model_name)
	# Identify words shared between vocabularies and create training data
	X_train = []
	y_train = []
	for i in range(input_vocabulary.count):
		w = input_vocabulary.vocabulary[i]
		other_idx = output_vocabulary.get_index(w)
		if other_idx:
			w_count = words.count(w)
			if w_count >= threshold_count:
				for x in range(w_count//threshold_count):
					X_train.append(input_embeddings[i])
					y_train.append(output_embeddings[other_idx])
	X_train = np.squeeze(np.asarray(X_train))
	y_train = np.squeeze(np.asarray(y_train))
	return X_train, y_train
	
def train_transform_model(input_model_name, output_model_name):
	# Create transform model
	transform_model = TransformModel(input_model_name, output_model_name, FLAGS.dimensions)
	transform_model.init_model()
	# Get training data
	X_train, y_train = get_transform_data(input_model_name, output_model_name)
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
	
def get_word_list(text):
	words = [w.lower() for w in text]
	words = sorted(set(words))
	words = [w for w in words if w not in "''``.,!?;:--()"]
	return words

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
	# Create replacement dictionary
	keys = get_word_list(text)
	r_dict = {}
	for word in keys:
		if word in input_vocabulary.vocabulary:
			print('"'+word+'" will be replaced with "', end="")
			# Transform to output embedding
			idx = input_vocabulary.get_index(word)
			embedding = input_embeddings[idx]
			transformed_embedding = transform_model.transform(embedding)
			# Find nearest neighbour in output space
			new_word, score = find_nearest_word(output_vocabulary, output_embeddings, transformed_embedding)
			print(new_word+'" ('+str(score)+')')
			r_dict[word] = new_word
	new_tokens = [None] * len(text)
	# For each word in input text
	for i in range(len(text)):
		word = text[i]
		# Find if it exists in replacement dictionary
		if word in r_dict.keys():
			word = r_dict[word]
		# Replace the word
		new_tokens[i] = word
	new_text = nltk.Text(new_tokens)
	if not output_file and not output_file == "":
		# Print new text
		for word in new_text:
			print(word, end=' ')
	else:
		text_file = open(output_file, "wb")
		for word in new_text:
			text_file.write((word + " ").encode('UTF-8'))
		text_file.close()
	
def main(argv):
	# Check that relevant parameters have been given
	if (len(sys.argv) < 2):
		print("Usage: python text-style-transfer.py <operation> [other arguments]")
		sys.exit()
	# Interpret command line arguments and call correct function
	operation = sys.argv[1]
	if FLAGS.debug: print("Dimensions: " + str(FLAGS.dimensions))
	if operation == "create-vocabulary":
		model_name = sys.argv[2]
		create_vocabulary(model_name)
	elif operation == "train-transform":
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
		proximity_test(model_name)
	
if __name__== "__main__":
	app.run(main)
