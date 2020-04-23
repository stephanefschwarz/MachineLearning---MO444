from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def generate_padded_docs(vocab_size=5000, vector_size=500, documents_as_list):

	encoded_docs = [one_hot(doc, vocab_size) for doc in documents_as_list]
	padded_docs = pad_sequences(encoded_docs, maxlen=vector_size, padding='post')

	return padded_docs

