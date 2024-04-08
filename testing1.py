# import spacy

# # Load spaCy English language model
# nlp = spacy.load("en_core_web_sm")

# # Example text to process
# text = "G2 has helped our customers publicly validate us to prospects and has helped us build pipeline and be considered for opportunities where we likely would not have been found otherwise. Also it has aided as a great validation point for customers in meetings and demo's as to our market profile and position.  It has helped us need less reference calls and enabled our customers to have a more open and valid place to voice their opinion on us."

# # Process the text using spaCy NLP pipeline
# doc = nlp(text)

# # Define function to extract token features
# def extract_token_features(token):
#     features = {
#         "text": token.text,
#         "lemma": token.lemma_,
#         "pos": token.pos_,
#         "dep": token.dep_,
#         "is_stop": token.is_stop,
#         "is_alpha": token.is_alpha
#     }
#     return features

# # Extract token-level features for each token in the processed document
# token_features = [extract_token_features(token) for token in doc]

# # Print token-level features
# print("Token Features:")
# for i, token in enumerate(doc):
#     print(f"Token {i + 1}: {token.text} | Lemma: {token.lemma_} | POS: {token.pos_} | Dependency: {token.dep_} | Stop Word: {token.is_stop} | Alphabetic: {token.is_alpha}")

# # Print extracted token features
# print("\nExtracted Token Features:")
# for i, feature in enumerate(token_features):
#     print(f"Token {i + 1}: {feature}")

# # Example of combining multiple features into a representation
# combined_features = [(token.text, token.pos_, token.dep_) for token in doc]
# print("\nCombined Features:")
# print(combined_features)

# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from nltk.tokenize import word_tokenize

# # Download NLTK resources (if not already downloaded)
# nltk.download('punkt')
# nltk.download('vader_lexicon')

# def analyze_sentiment(paragraph):
#     # Tokenize the paragraph into words
#     words = word_tokenize(paragraph)
    
#     # Initialize sentiment analyzer
#     analyzer = SentimentIntensityAnalyzer()
    
#     # Initialize lists to store good and bad features
#     good_features = []
#     bad_features = []
    
#     # Analyze sentiment for each word
#     for word in words:
#         sentiment_score = analyzer.polarity_scores(word)['compound']
        
#         # Classify word as good or bad based on sentiment score
#         if sentiment_score >= 0.2:
#             good_features.append(word)
#         elif sentiment_score <= -0.2:
#             bad_features.append(word)
    
#     return good_features, bad_features

# # Example paragraph
# paragraph = "The end to end service and process has been the highlight to my experience. Advocately has ensured our success by assisting us with implementing our review campaigns. They've consistently optimised our process to achieve our goals. There is little that I dislike. Understanding exactly what each of the features means and the flow between them has been my only slight hindrance. If you're a Saas company focused on growth, you need Advocately in your sales and marketing stack.We've been able to effectively scale our review processes by encouraging our promoters to share their experience on various review websites. Previously this was a very manual exercise to connect and remind our promoters that we'd love their support. Now with Adovately we can politely reach out and explain how they could really help with our growth success by leaving a review of their experience."

# # Perform sentiment analysis and extract features
# good_features, bad_features = analyze_sentiment(paragraph)

# # Print the segregated good and bad features
# print("Good Features:", good_features)
# print("Bad Features:", bad_features)


# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from nltk.util import ngrams
# from collections import defaultdict

# # Download NLTK resources (if not already downloaded)
# nltk.download('punkt')
# nltk.download('vader_lexicon')
# nltk.download('stopwords')

# def extract_key_points(sentences, sentiment_score):
#     # Initialize list to store key points
#     key_points = []
    
#     # Process each sentence to extract key phrases based on sentiment score
#     for sentence in sentences:
#         words = word_tokenize(sentence.lower())  # Convert to lowercase
        
#         # Filter out stopwords and punctuation
#         filtered_words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
        
#         # Extract bigrams and trigrams
#         bigrams = list(ngrams(filtered_words, 2))
#         trigrams = list(ngrams(filtered_words, 3))
        
#         # Determine key points based on sentiment score
#         if sentiment_score >= 0.2:
#             if bigrams:
#                 key_points.append(' '.join(bigrams[0]))  # Use the first bigram as a key point
#         elif sentiment_score <= -0.2:
#             if bigrams:
#                 key_points.append(' '.join(bigrams[0]))  # Use the first bigram as a key point
    
#     return key_points

# def analyze_sentiment_with_key_points(paragraph):
#     # Tokenize the paragraph into sentences
#     sentences = sent_tokenize(paragraph)
    
#     # Initialize sentiment analyzer
#     analyzer = SentimentIntensityAnalyzer()
    
#     # Initialize dictionary to store good and bad points with associated key points
#     good_points = defaultdict(list)
#     bad_points = defaultdict(list)
    
#     # Analyze sentiment for each sentence
#     for sentence in sentences:
#         sentiment_score = analyzer.polarity_scores(sentence)['compound']
        
#         # Extract key points from the sentence based on sentiment score
#         key_points = extract_key_points([sentence], sentiment_score)
        
#         # Classify sentence as good or bad based on sentiment score
#         if sentiment_score >= 0.2:
#             if key_points:
#                 good_points[key_points[0]].append(sentence)  # Use the first key point as a good point
#         elif sentiment_score <= -0.2:
#             if key_points:
#                 bad_points[key_points[0]].append(sentence)  # Use the first key point as a bad point
    
#     return good_points, bad_points

# # Example paragraph
# paragraph = """
# The end to end service and process has been the highlight to my experience. Advocately has ensured our success by assisting us with implementing our review campaigns. 
# They've consistently optimized our process to achieve our goals. There is little that I dislike. Understanding exactly what each of the features means and the flow between them has been my only slight hindrance. 
# If you're a SaaS company focused on growth, you need Advocately in your sales and marketing stack. We've been able to effectively scale our review processes by encouraging our promoters to share their experience on various review websites. 
# Previously this was a very manual exercise to connect and remind our promoters that we'd love their support. Now with Advocately we can politely reach out and explain how they could really help with our growth success by leaving a review of their experience.
# """

# # Perform sentiment analysis and extract key points for good and bad points
# good_points, bad_points = analyze_sentiment_with_key_points(paragraph)

# # Print the segregated good points with associated key points and sentences
# print("Good Points:")
# for key_point, sentences in good_points.items():
#     print(f"- {key_point}: {', '.join(sentences[:2])}")  # Print one or two sentences for each good point

# print()

# # Print the segregated bad points with associated key points and sentences
# print("Bad Points:")
# for key_point, sentences in bad_points.items():
#     print(f"- {key_point}: {', '.join(sentences[:2])}")  # Print one or two sentences for each bad point

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import defaultdict

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

def extract_phrases(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize phrase extractor
    phrases = []
    
    # Process each sentence to extract meaningful phrases (phrases with 4 or more words)
    for sentence in sentences:
        words = word_tokenize(sentence.lower())  # Convert to lowercase
        
        # Filter out stopwords and punctuation
        filtered_words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
        
        # Extract phrases with 4 or more words
        if len(filtered_words) >= 4:
            phrases.append(' '.join(filtered_words))
    
    return phrases

def analyze_sentiment_with_phrases(paragraph):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    
    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Initialize dictionary to store good and bad points with associated phrases
    good_points = defaultdict(list)
    bad_points = defaultdict(list)
    
    # Analyze sentiment for each sentence
    for sentence in sentences:
        sentiment_score = analyzer.polarity_scores(sentence)['compound']
        
        # Extract meaningful phrases from the sentence
        phrases = extract_phrases(sentence)
        
        # Classify sentence as good or bad based on sentiment score
        if sentiment_score >= 0.2:
            for phrase in phrases:
                good_points[phrase].append(sentence)
        elif sentiment_score <= -0.2:
            for phrase in phrases:
                bad_points[phrase].append(sentence)
    
    return good_points, bad_points

# Example paragraph
paragraph = """
The end to end service and process has been the highlight to my experience. Advocately has ensured our success by assisting us with implementing our review campaigns. 
They've consistently optimized our process to achieve our goals. There is little that I dislike. Understanding exactly what each of the features means and the flow between them has been my only slight hindrance. 
If you're a SaaS company focused on growth, you need Advocately in your sales and marketing stack. We've been able to effectively scale our review processes by encouraging our promoters to share their experience on various review websites. 
Previously this was a very manual exercise to connect and remind our promoters that we'd love their support. Now with Advocately we can politely reach out and explain how they could really help with our growth success by leaving a review of their experience.
"""

# Perform sentiment analysis and extract detailed phrases for good and bad points
good_points, bad_points = analyze_sentiment_with_phrases(paragraph)

# Print the segregated good points with associated phrases
print("Good Points:")
for phrase, sentences in good_points.items():
    print(f"- {phrase}: {', '.join(sentences)}")

print()

# Print the segregated bad points with associated phrases
print("Bad Points:")
for phrase, sentences in bad_points.items():
    print(f"- {phrase}: {', '.join(sentences)}")

