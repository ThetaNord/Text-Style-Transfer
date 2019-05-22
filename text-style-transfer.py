import sys

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

import nltk
#nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords

# Class for training word2vec embeddings
class EmbeddingModel:

	def __init__(self, n=100):
		self.model = create_model(n)
		self.dimensions = n

	def create_model(self, n):
		model = None
		
		return model
		
# Model for transforms between to word2vec embeddings (order matters)
class TransformModel:
	
	def __init__(self):
		self.input_name = "name"
		self.output_name = "name"
	
		
def train_embedding(source_path, model_name, epochs=50):
	# Load source file
	sourcefile = open(source_path).read()
	# Preprocess source file (removing/replacing uncommon words?)
	raw = sourcefile
	#raw = raw.replace('"',"")
	tokens = word_tokenize(raw)
	text = nltk.Text(tokens)
	print(text[:20])
	# Create vocabulary
	words = [w.lower() for w in text]
	vocab = sorted(set(words))
	print(len(vocab))
	print(vocab[:50])
	# Create skip-grams
	# Create model
	# Train model
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
		print("Usage: python text-style-transfer.py <source_path>")
		sys.exit()
	# Interpret command line argument
	source_path = sys.argv[1]
	print('ú')
	print("úú")
	print('ú'.upper())
	print(u'ú'.lower())
	# Call correct function
	train_embedding(source_path, model_name="model")
	
if __name__== "__main__":
	main()
