import re
import emoji

class Patterns:

	URL = re.compile('((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+ | (?:www\. | [\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)')

	AT = re.compile('@[a-zA-Z]+')

	HASH = re.compile('#[a-zA-Z]+')

	NUM = re.compile('[0-9]+')

	SMILEYS = re.compile(r'(?:X | : | ; | =)(?:-)?(?:\) | \( | O | D | P | S){1,}', re.IGNORECASE)

	file_object  = open("emoji.txt", "r")
	st = file_object.readlines()

	
	#EMOJIS = re.compile(''.join(st))

	#EMOJIS =   re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

	
	EMOJIS = re.compile("["
		"U+E049"
		u"\U00010000-\U0010ffff"
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002600-\U000027BF"
        u"\U0001f300-\U0001f64F"
        u"\U0001f680-\U0001f6FF"
        u"\u2600-\u27BF"
        u"\uD83C"
        u"\uDF00-\uDFFF"
        u"\uD83D"
        u"\uDC00-\uDE4F"
        u"\uD83D"
        u"\uDE80-\uDEFF"
                           "]+", flags=re.UNICODE)
   