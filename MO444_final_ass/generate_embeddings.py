import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

def get_pretrained_model():

	return KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def get_embeddings(tokens_list, vector, generate_missing=True, dim=300):

	if (len(tokens_list) < 1):        
        return np.zeros(dim)

    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(dim) for word in tokens_list]

    else:
        vectorized = [vector[word] if word in vector else np.zeros(dim) for word in tokens_list]

    length = len(vectorized)
    summed = np.sum(vectorized, axis = 0)
    avg = np.divide(summed, length)
    
    return avg

def get_embeddings_avg(vectors, clean_tokens, generate_missing=True):

	embeddings = clean_tokens.apply(lambda x: get_avg_word2vec(x, vectors, generate_missing=generate_missing))
    
    return list(embeddings)

