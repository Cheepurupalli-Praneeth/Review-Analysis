import json

# Load the JSON data
with open('data1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Function to concatenate comment answers into a single paragraph
def concatenate_comments(comment_answers):
    return ' '.join(answer['value'] for answer in comment_answers.values())

# Iterate over each survey response
for response in data['data']:
    attributes = response['attributes']
    relationships = response['relationships']

    # Extract relevant attributes
    response_id = response['id']
    product_name = attributes['product_name']
    star_rating = attributes['star_rating']
    title = attributes['title']
    comment_answers = attributes['comment_answers']
    secondary_answers = attributes['secondary_answers']

    # Concatenate comment answers into a single paragraph
    comment_feedback = concatenate_comments(comment_answers)

    # Extract secondary answers details
    secondary_details = []
    for key, value in secondary_answers.items():
        secondary_details.append(f"{value['text']}: {value['value']}")

    # Format and print the concise summary
    print(f"ID: {response_id}")
    print(f"Product Name: {product_name}")
    print(f"Star Rating: {star_rating}")
    print(f"Title: {title}")
    print(f"Comment Feedback: {comment_feedback}")
    print("Secondary Answers:")
    for detail in secondary_details:
        print(f"  - {detail}")
    print()  # Print an empty line for readability
