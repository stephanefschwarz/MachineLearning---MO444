from sklearn.feature_extraction.text import TfidfVectorizer

def set_tfidf(analyzer="word", ngram_range=(1,1), stop_words="english", lowercase=True, max_features=500, use_idf=True):

	tfidf = TfidfVectorizer(analyzer="word", ngram_range=(4,4), stop_words="english", lowercase=True, max_features=500, use_idf=True)
	
	return tfidf

def generate_tfidf(tfidf, raw_documents):
	
	tfidf_vector = tfidf.fit_transform(raw_documents)
	
	return tfidf_vector
	



