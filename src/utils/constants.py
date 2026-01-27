from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)

# Create a custom list of "noise words" specific to Recruitment
JD_STOP_WORDS = list(vectorizer.get_stop_words()) + [
    'experience', 'knowledge', 'skills', 'work', 'team', 'working', 
    'paid', 'days', 'requirements', 'looking', 'company', 'development', 
    'software', 'opportunity', 'years', 'plus', 'business', 'strong', 
    'good', 'understanding', 'proficiency', 'based', 'using', 'english'
]