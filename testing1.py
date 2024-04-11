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


#  
# segregate sentences
# 
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # Initialize VADER sentiment analyzer
# analyzer = SentimentIntensityAnalyzer()

# # Define the input text (your provided paragraph)
# text = """
# Our onboarding specialist and account manager are always quick to respond and answer all of our questions. They made it extremely easy to increase our review volume.  Currently, I don't have anything that I dislike. The analytics could be slightly improved (as far as deeper customization) but it works as is. From a customer marketing position, I'm looking to increase our positive online reputation. G2 is helping that along.
# """

# # Split text into sentences
# sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]

# # Function to predict sentiment using VADER
# def predict_sentiment_vader(sentence):
#     sentiment_score = analyzer.polarity_scores(sentence)
#     compound_score = sentiment_score['compound']
    
#     # Classify sentiment based on compound score
#     if compound_score >= 0.05:
#         return 'Positive'
#     elif compound_score <= -0.05:
#         return 'Negative'
#     else:
#         return 'Neutral'  # You can adjust this threshold based on your needs

# # Predict sentiment for each sentence using VADER
# sentiment_results_vader = {sentence: predict_sentiment_vader(sentence) for sentence in sentences}

# # Separate positive and negative sentences
# positive_sentences_vader = [sentence for sentence, sentiment in sentiment_results_vader.items() if sentiment == 'Positive']
# negative_sentences_vader = [sentence for sentence, sentiment in sentiment_results_vader.items() if sentiment == 'Negative']

# # Print segregated sentences
# print("Positive Sentences (VADER):")
# for sentence in positive_sentences_vader:
#     print("-", sentence)

# print("\nNegative Sentences (VADER):")
# for sentence in negative_sentences_vader:
#     print("-", sentence)

# from transformers import pipeline

# # Sentiment analysis pipeline
# sentiment_analysis = pipeline("sentiment-analysis")

# # Text summarization pipeline (requires additional library installation)
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Install required libraries (ensure they are installed correctly)
# # !pip install transformers rake-nltk

# # Specify summarization model (you can experiment with different models)
# model_name = "facebook/bart-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # NER model and tokenizer (using the public dslim/bert-base-NER model)
# ner_model_name = "dslim/bert-base-NER"
# ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
# ner_model = pipeline("ner")

# # Keyword extraction object
# r = Rake()

# # Your text (replace with your actual text)
# text = """The Customer Success Team was incredibly helpful. We saw immediate ROI thanks to their Review campaign. 
# Growing companies often have new employees finding their way, but a strong core team helps guide them. 
# G2 Crowd is expanding our brand awareness. 
# For organizations like BuyerQuest, competing against giants requires user advocacy."""

# # Split the text into sentences
# sentences = text.split(". ")

# # Sentiment analysis and results
# sentiment_results = []
# for sentence in sentences:
#   sentiment = sentiment_analysis(sentence)
#   sentiment_results.append(sentiment[0])

# # Sentiment classification function
# def classify_sentiment(sentiment):
#   if sentiment['label'] == 'POSITIVE':
#     return 'positive'
#   elif sentiment['label'] == 'NEGATIVE':
#     return 'negative'
#   else:
#     return 'neutral'

# # Function to summarize sentences using pre-trained model, aiming for a review-like style
# def summarize_to_review(sentence):
#   # Tokenize the sentence
#   inputs = tokenizer(sentence, return_tensors="pt")

#   # Generate summary using beam search
#   output = model.generate(**inputs, max_length=10, num_beams=3)  # Adjust max_length for concise summaries

#   # Decode the generated summary tokens
#   summary = tokenizer.decode(output[0], skip_special_tokens=True)

#   # Perform NER on the sentence
#   ner_results = ner_model(sentence)

#   # Identify relevant entities
#   relevant_entities = [entity['word'] for entity in ner_results if entity['entity_group'] in ['ORG', 'PER']]

#   # Summarize based on sentiment and entities (avoid hardcoded phrases)
#   if classify_sentiment(sentiment_results[sentences.index(sentence)]) == 'positive':
#     if relevant_entities:
#       summary = f"{relevant_entities[0]} {summary}"  # Focus on entity and relevant aspect from summary
#     else:
#       summary = f"{summary} was a great experience."  # Generic positive feedback
#   else:
#     summary = f"{summary} needs improvement."  # Generic negative feedback

#   return summary

# # Segregate and summarize sentences
# positive_sentences = []
# negative_sentences = []
# for i, sentence in enumerate(sentences):
#   sentiment = classify_sentiment(sentiment_results[i])
#   if sentiment == 'positive':
#     positive_sentences.append(summarize_to_review(sentence))
#   elif sentiment == 'negative':
#     negative_sentences.append(summarize_to_review(sentence))

# # Print results
# print("Positive aspects (like a review):")
# for sentence in positive_sentences:
#   print(sentence)

# print("\nChallenges mentioned:")
# for sentence in negative_sentences:
#   print(sentence)

# 
# down code works ok ok
# 


# import nltk
# from nltk.tokenize import sent_tokenize
# from textblob import TextBlob
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Download NLTK resources if needed
# nltk.download('punkt')

# # Function to preprocess text
# def preprocess(text):
#   sentences = sent_tokenize(text)
#   return sentences

# # Function to get sentiment
# def get_sentiment(sentence):
#   sentiment = TextBlob(sentence).sentiment
#   if sentiment.polarity > 0:
#     return "positive"
#   elif sentiment.polarity < 0:
#     return "negative"
#   else:
#     return "neutral"

# # Function to summarize with chosen model (replace 't5-base/summarization' if needed)
# def summarize(model_name, sentence, access_token):
#   try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token)

#     input_ids = tokenizer.encode(sentence, return_tensors="pt")
#     output_ids = model.generate(input_ids, num_beams=15)  # Use beam search for diverse summaries
#     summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return summary
#   except Exception as e:
#     print(f"Error loading model or generating summary: {e}")
#     return None

# # Replace with your actual access token (important: keep it secure)
# access_token = "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"

# # Example usage
# text = "some options that are available in our plan help us find new leads. Also, the reviews written by our clients add more credibility to the company, especially the badges that we are using on our website and in our promo spaces, in presentations and other ads visuals. I do not like that the information about the written and rejected reviews is hidden. There have been several instances where reviews written by our clients have been rejected for no good reason. In case there are controversial issues with the review, it would be better to transparently resolve them directly with the product team.\r\nThe market Presence Scoring algorithm is a bit 'blurry'. We can't directly influence this and your position on the G2 grid.\r\nVery long questionnaire, please make it lightweight because it is difficult to get customers to go through all the pages of the questionnaire in order for the review to be qualified for the report.\r\nAlso, make screenshot a required field, because it's optional now and clients don't add it, but it's a reason for you to reject the review. We are in a highly competitive market, it's important for us to better understand our position in the marketplace of employee intranets and communications.\r\nIt's helped drive new leads while providing us with the tools to gain new insights into our visitor audience.\r\nQuarterly reports and badges help us build trust amongst new leads and across different marketing channels."

# sentences = preprocess(text)

# for sentence in sentences:
#   sentiment = get_sentiment(sentence)
#   if sentiment != "neutral":
#     # Choose the model you want to use for summarization (replace 't5-base/summarization' if needed)
#     model_name = "t5-base"
#     summary = summarize(model_name, sentence, access_token)
#     if summary:
#       print(f"{sentiment.upper()}: {summary}")
#     else:
#       print(f"{sentiment.upper()}: Error generating summary")


# import nltk
# from nltk.tokenize import sent_tokenize
# from textblob import TextBlob
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Download NLTK resources if needed
# nltk.download('punkt')

# # Function to preprocess text
# def preprocess(text):
#   sentences = sent_tokenize(text)
#   return sentences

# # Function to get sentiment (optional)
# def get_sentiment(sentence):
#   try:
#     sentiment = TextBlob(sentence).sentiment
#     if sentiment.polarity > 0:
#       return "positive"
#     elif sentiment.polarity < 0:
#       return "negative"
#     else:
#       return "neutral"
#   except Exception as e:
#     print(f"Error performing sentiment analysis: {e}")
#     return None  # Indicate sentiment analysis error

# # Function to summarize with T5-large
# def summarize(model_name, sentence, access_token=None):
#   try:
#     if access_token:
#       tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
#       model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token)
#     else:
#       tokenizer = AutoTokenizer.from_pretrained(model_name)
#       model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     input_ids = tokenizer.encode(sentence, return_tensors="pt")
#     output_ids = model.generate(input_ids, num_beams=5)  # Use beam search for diverse summaries
#     summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return summary
#   except Exception as e:
#     print(f"Error loading model or generating summary: {e}")
#     return None

# # Replace with your actual access token (important: keep it secure)
# access_token = "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"  # Only needed for private models

# # Example usage
# text = "some options that are available in our plan help us find new leads. Also, the reviews written by our clients add more credibility to the company, especially the badges that we are using on our website and in our promo spaces, in presentations and other ads visuals. I do not like that the information about the written and rejected reviews is hidden. There have been several instances where reviews written by our clients have been rejected for no good reason. In case there are controversial issues with the review, it would be better to transparently resolve them directly with the product team.\r\nThe market Presence Scoring algorithm is a bit 'blurry'. We can't directly influence this and your position on the G2 grid.\r\nVery long questionnaire, please make it lightweight because it is difficult to get customers to go through all the pages of the questionnaire in order for the review to be qualified for the report.\r\nAlso, make screenshot a required field, because it's optional now and clients don't add it, but it's a reason for you to reject the review. We are in a highly competitive market, it's important for us to better understand our position in the marketplace of employee intranets and communications.\r\nIt's helped drive new leads while providing us with the tools to gain new insights into our visitor audience.\r\nQuarterly reports and badges help us build trust amongst new leads and across different marketing channels."

# sentences = preprocess(text)

# for sentence in sentences:
#   # Optional sentiment analysis (uncomment to enable)
#   # sentiment = get_sentiment(sentence)

#   # Choose the model (replace if desired)
#   model_name = "t5-large"

#   summary = summarize(model_name, sentence, access_token)
#   if summary:
#     # Print sentiment if enabled, otherwise just print summary
#     # if sentiment:
#     #   print(f"{sentiment.upper()}: {summary}")
#     # else:
#     print(f"{summary}")
#   else:
#     print(f"Error generating summary")



# import nltk
# from nltk.tokenize import sent_tokenize
# from textblob import TextBlob
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Download NLTK resources if needed
# nltk.download('punkt')

# # Function to preprocess text
# def preprocess(text):
#   sentences = sent_tokenize(text)
#   return sentences

# # Function to get sentiment
# def get_sentiment(sentence):
#   sentiment = TextBlob(sentence).sentiment
#   if sentiment.polarity > 0:
#     return "positive"
#   elif sentiment.polarity < 0:
#     return "negative"
#   else:
#     return "neutral"

# # Function to summarize with chosen model
# def summarize(model_name, sentence, access_token):
#   try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     input_ids = tokenizer.encode(sentence, return_tensors="pt")
#     output_ids = model.generate(input_ids, num_beams=10)  # Use beam search for diverse summaries
#     summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return summary
#   except Exception as e:
#     print(f"Error loading model or generating summary: {e}")
#     return None

# # Replace with your actual access token (important: keep it secure)
# access_token = "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"  # Only needed for private models

# # Example usage
# text = "some options that are available in our plan help us find new leads. Also, the reviews written by our clients add more credibility to the company, especially the badges that we are using on our website and in our promo spaces, in presentations and other ads visuals. I do not like that the information about the written and rejected reviews is hidden. There have been several instances where reviews written by our clients have been rejected for no good reason. In case there are controversial issues with the review, it would be better to transparently resolve them directly with the product team.\r\nThe market Presence Scoring algorithm is a bit 'blurry'. We can't directly influence this and your position on the G2 grid.\r\nVery long questionnaire, please make it lightweight because it is difficult to get customers to go through all the pages of the questionnaire in order for the review to be qualified for the report.\r\nAlso, make screenshot a required field, because it's optional now and clients don't add it, but it's a reason for you to reject the review. We are in a highly competitive market, it's important for us to better understand our position in the marketplace of employee intranets and communications.\r\nIt's helped drive new leads while providing us with the tools to gain new insights into our visitor audience.\r\nQuarterly reports and badges help us build trust amongst new leads and across different marketing channels."

# sentences = preprocess(text)

# for sentence in sentences:
#   sentiment = get_sentiment(sentence)
#   if sentiment != "neutral":
#     # Choose the model you want to use for summarization (replace 'facebook/bart-base' if needed)
#     model_name = "facebook/bart-base"
#     summary = summarize(model_name, sentence, access_token)
#     if summary:
#       print(f"{sentiment.upper()}: {summary}")
#     else:
#       print(f"{sentiment.upper()}: Error generating summary")


#this is good


# from google.cloud import aiplatform
# import os

# # Replace with your Google Cloud project ID
# project_id = "your-project-id"

# # Set the region where your project resides
# endpoint = f"projects/{project_id}/locations/us-central1"  # Replace with your region

# # Set up authentication (use either Application Default Credentials or Service Account)
# # 1. Application Default Credentials (recommended)
# #  - Ensure your code runs in an environment with credentials (e.g., Google Compute Engine)

# # 2. Service Account Key (for more control)
# #  - Create a service account and download its key file (JSON format)
# #  - Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the key file path:
# #    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"

# from google.cloud import aiplatform
# import os

# # Replace with your Google Cloud project ID
# project_id = "your-project-id"

# # Set the region where your project resides
# endpoint = f"projects/{project_id}/locations/us-central1"  # Replace with your region

# # Set up authentication (use either Application Default Credentials or Service Account)
# # 1. Application Default Credentials (recommended)
# #  - Ensure your code runs in an environment with credentials (e.g., Google Compute Engine)

# # 2. Service Account Key (for more control)
# #  - Create a service account and download its key file (JSON format)
# #  - Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the key file path:
# #    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"

# from google.cloud import aiplatform
# import os

# # Replace with your Google Cloud project ID
# project_id = "titanium-acumen-419818"

# # Set the region where your project resides
# endpoint = f"projects/{project_id}/locations/us-central1"  # Replace with your region

# # Set up authentication (use either Application Default Credentials or Service Account)
# # 1. Application Default Credentials (recommended)
# #  - Ensure your code runs in an environment with credentials (e.g., Google Compute Engine)

# # 2. Service Account Key (for more control)
# #  - Create a service account and download its key file (JSON format)
# #  - Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the key file path:
# #    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"

# client_options = {"api_endpoint": endpoint}
# client = aiplatform.gapic.TextServiceClient(client_options=client_options)


# def generate_text(text_input, prompt, max_tokens=150, temperature=0.7):
#   """
#   Generates text using Google Cloud Natural Language API for text generation.

#   Args:
#       text_input: The initial text input for the prompt.
#       prompt: The prompt to guide the text generation.
#       max_tokens: Maximum number of tokens to generate (default: 150).
#       temperature: Controls randomness of the generated text (0: deterministic, 1: maximal, default: 0.7).

#   Returns:
#       The generated text based on the prompt.
#   """

#   # Concatenate text input and prompt
#   combined_text = text_input + prompt

#   document = aiplatform.types.Document(
#       content=combined_text,
#       type_=aiplatform.enums.Document.Type.PLAIN_TEXT
#   )

#   # Customize features for text generation
#   features = aiplatform.types.AnalyzeTextRequest.Features(
#       generate_text=aiplatform.types.GenerateTextRequest(
#           max_length=max_tokens,
#           temperature=temperature
#       )
#   )

#   request = aiplatform.types.AnalyzeTextRequest(
#       document=document,
#       features=features
#   )

#   response = client.analyze_text(request=request)

#   # Extract generated text
#   generated_text = response.generate_text_result.generated_text

#   return generated_text

# # Example usage
# text_input = "We love the friendly, positive attitude of all the G2 employees we've encountered. The follow-ups and the problem-solving have been above and beyond. \r\nThe backend user experience of the platform is easy to navigate and simple to perform updates and changes of content etc. There is a lot going on, lots of new integrations, always lots of news - that sometimes it feels a bit overwhelming. Maybe some kind of a monthly news reel, like a quick roundup, could be produced and sent out instead of so much static content. When we hit walls because of company restrictions, we are consistently offered ways around it. That way, when any restrictions are lifted, we will be ready to take advantage."
# prompt = "Generate positive and negative feature sets"

# generated_text = generate_text(text_input, prompt)

# print(f"Combined text: {text_input}{generated_text}")
# from google.cloud import aiplatform

# # Replace with your project ID and region
# project_id = "titanium-acumen-419818"
# location = "asia-east2"

# # Initialize Vertex AI client
# aiplatform.init(project=project_id, location=location)

# # Define your textual data and prompt
# text_input = "We love the friendly, positive attitude of all the G2 employees we've encountered. The follow-ups and the problem-solving have been above and beyond. \r\nThe backend user experience of the platform is easy to navigate and simple to perform updates and changes of content etc. There is a lot going on, lots of new integrations, always lots of news - that sometimes it feels a bit overwhelming. Maybe some kind of a monthly news reel, like a quick roundup, could be produced and sent out instead of so much static content. When we hit walls because of company restrictions, we are consistently offered ways around it. That way, when any restrictions are lifted, we will be ready to take advantage."
# prompt = "Generate positive and negative feature sets"

# # Endpoint for your deployed model (obtain from Vertex AI UI)
# endpoint = "projects/titanium-acumen-419818/logs/cloudaudit.googleapis.com%2Factivity"

# # Use the appropriate prediction method based on your model type
# # (classification, summarization, generation)
# response = aiplatform.Endpoint.predict(endpoint, instances=[text_input + prompt])


# # Process the prediction response based on your model output format
# print(response.predictions)
 