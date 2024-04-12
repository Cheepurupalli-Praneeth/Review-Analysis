import requests
import json
from ctransformers import AutoModelForCausalLM

def load_api_key():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        return config.get('access_token', None)


BASE_URL = "https://data.g2.com/api/v1/survey-responses"
HEADERS = {
    "Authorization": f"Token token={load_api_key()}",
    "Content-Type": "application/vnd.api+json"
}

#loading slm model
SLM = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-7B-Chat-GGML', model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin')



def process_and_summarize_response(response):
    """
    Process a survey response and summarize key information.

    Args: Survey response data.

    Returns: Summarized response data.
    """
    attributes = response['attributes']
    response_id = response['id']
    product_name = attributes['product_name']
    if product_name == "G2 Marketing Solutions":
        star_rating = attributes['star_rating']
        title = attributes['title']
        comment_answers = attributes['comment_answers']
        secondary_answers = attributes['secondary_answers']

        # Concatenate all comment answers into a single feedback string
        comment_feedback = ' '.join(answer['value'] for answer in comment_answers.values())

        # Use the language model to generate feature sets
        generated_features = generate_features(comment_feedback)

        # Format secondary answers
        secondary_details = [{value['text']: value['value']} for value in secondary_answers.values()]

        # Create summarized response dictionary
        summarized_response = {
            "ID": response_id,
            "Product Name": product_name,
            "Star Rating": star_rating,
            "Title": title,
            "Comment Feedback": comment_feedback,
            "Generated Features": generated_features,
            "Secondary Answers": secondary_details
        }

        return summarized_response
    else:
        return None

def generate_features(comment_feedback):
    """
    Generate feature sets using the language model.

    Args: Concatenated comment feedback.

    Returns:Generated feature sets.
    """
    generated_text = ''
    for word in SLM('For the following review, generate feature sets, in points, customers are looking for(maximum 5 words per point).'+comment_feedback, stream=True):
        generated_text += word
    return generated_text

    # generated_text=''.join(LLM('For the following review, generate feature sets(maximum 5 words per point) customers are looking for.'+comment_feedback, stream=True))
    # return generated_text

def fetch_survey_responses(base_url, headers, start_page_number=1):
    """
    Fetch survey responses from the G2 API and process them.

    Args: Base URL for API endpoint, Request headers, Starting page number. Defaults to 1.
    """
    page_number = start_page_number

    while True:
        url = f"{base_url}?page%5Bnumber%5D={page_number}&page%5Bsize%5D=100"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            for survey_response in data['data']:
                # print(survey_response)
                summarized_response = process_and_summarize_response(survey_response)
                if summarized_response is not None:
                    print(json.dumps(summarized_response, ensure_ascii=False, indent=4))
            if 'next' in data['links']:
                page_number += 1
            else:
                print("No more 'next' link found. Pagination complete.")
                break
        else:
            print(f"Error occurred for page {page_number}: Status Code {response.status_code}")
            break


def main():
    fetch_survey_responses(BASE_URL, HEADERS)

if __name__ == "__main__":
    main()
