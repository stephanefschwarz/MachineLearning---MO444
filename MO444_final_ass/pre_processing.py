#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from googletrans import Translator
from utils import Patterns
import preprocessor as p


def url_to_tag(document):

	return re.sub(Patterns.URL, u'<URL>', document)

def at_to_tag(document):

	return re.sub(Patterns.AT, u'<AT>', document)

def hash_to_tag(document):

	return re.sub(Patterns.HASH, u'<HASH>', document)	

def number_to_tag(document):

	return re.sub(Patterns.NUM, u'<NUM>', document)

def emoji_to_tag(document):

	return re.sub(Patterns.EMOJIS, u'<EMO>', document)

def smileys_to_tag(document):

	return re.sub(Patterns.SMILEYS, u'<SMI>', document)

def translate_document(document):

	translator = Translator()
	
	try:
		translated_document = translator.translate(document, dest='en')

	except ValueError:

		translated_document = document

	return translated_document

def pre_processing_text(post):

	post = emoji_to_tag(post)
	post = smileys_to_tag(post)
	post = url_to_tag(post)
	post = at_to_tag(post)
	post = hash_to_tag(post)
	post = number_to_tag(post)
	return translate_document(post)

if __name__ == '__main__':

	dataframe = pd.read_csv('new_fakenews_dataframe.csv')

	posts = dataframe['tweetText']

	label = dataframe['label']
	clean_posts = []
	i = 0
	for post in posts:
		print(i)
		i += 1 
		processed_post = translate_document(post)
		tp = type(processed_post)

		print(processed_post)		
		
		clean_posts.append(processed_post.text)

	
	processed_posts = pd.DataFrame({"tweetText":clean_posts})

	print(processed_posts)

	fakenews_dataframe = pd.concat([processed_posts["tweetText"], label], axis=1)

	fakenews_dataframe.to_csv("fakenews.csv")