from sklearn.metrics import (f1_score, accuracy_score)
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
import sklearn.ensemble

def random_forest(vector_train, label_train, n_estimators=38):
	
	x_train, x_test, y_train, y_test = train_test_split(vector_train, label_train, test_size=0.40, random_state=42)

	rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 38, n_jobs = 6) #38
	rf.fit(x_train, y_train)

	predictions = rf.predict(x_test)

	f1 = f1_score(y_test, predictions, average="weighted")
	accuracy = accuracy_score(y_test, predictions, normalize=True)
	
	return f1, accuracy

def naive_bayes(vector_train, label_train):

	x_train, x_test, y_train, y_test = train_test_split(vector_train, label_train, test_size=0.40, random_state=42)

	nb = GaussianNB()
	nb.fit(x_train, y_train)

	predictions = nb.predict(x_test)

	f1 = f1_score(y_test, predictions, average="weighted")
	accuracy = accuracy_score(y_test, predictions, normalize=True)
	
	return f1, accuracy

def neural_network(padded_docs, label, vocab_size, vector_size, epochs=20):

	x_train, x_test, y_train, y_test = train_test_split(padded_docs, label, test_size=0.40, random_state=42)

	model = Sequential()
	model.add(Embedding(vocab_size, 8, input_length=max_len))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])

	model.fit(x_train, y_train, epochs=epochs, verbose=0)

	predictions = model.predict(x_test)

	f1 = f1_score(y_test, predictions, average="weighted")
	accuracy = accuracy_score(y_test, predictions, normalize=True)
	
	return f1, accuracy
