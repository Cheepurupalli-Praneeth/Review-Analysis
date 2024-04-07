import spacy

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Example text to process
text = "The end to end service and process has been the highlight to my experience. Advocately has ensured our success by assisting us with implementing our review campaigns. They've consistently optimised our process to achieve our goals."

# Process the text using spaCy NLP pipeline
doc = nlp(text)

# Define function to extract token features
def extract_token_features(token):
    features = {
        "text": token.text,
        "lemma": token.lemma_,
        "pos": token.pos_,
        "dep": token.dep_,
        "is_stop": token.is_stop,
        "is_alpha": token.is_alpha
    }
    return features

# Extract token-level features for each token in the processed document
token_features = [extract_token_features(token) for token in doc]

# Print token-level features
print("Token Features:")
for i, token in enumerate(doc):
    print(f"Token {i + 1}: {token.text} | Lemma: {token.lemma_} | POS: {token.pos_} | Dependency: {token.dep_} | Stop Word: {token.is_stop} | Alphabetic: {token.is_alpha}")

# Print extracted token features
print("\nExtracted Token Features:")
for i, feature in enumerate(token_features):
    print(f"Token {i + 1}: {feature}")

# Example of combining multiple features into a representation
combined_features = [(token.text, token.pos_, token.dep_) for token in doc]
print("\nCombined Features:")
print(combined_features)
