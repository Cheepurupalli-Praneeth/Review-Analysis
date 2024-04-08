import requests
import json

base_url = "https://data.g2.com/api/v1/survey-responses"

headers = {
    "Authorization": "Token token=c1725e36612f450e1e0ae0b656a65f691d3d98d41308ba54059b28f32ecb9080",
    "Content-Type": "application/vnd.api+json"
}

def concatenate_comments(comment_answers):
    return ' '.join(answer['value'] for answer in comment_answers.values())

def process_and_summarize_response(response):
    attributes = response['attributes']
    response_id = response['id']
    product_name = attributes['product_name']
    star_rating = attributes['star_rating']
    title = attributes['title']
    comment_answers = attributes['comment_answers']
    secondary_answers = attributes['secondary_answers']

    comment_feedback = concatenate_comments(comment_answers)

    secondary_details = []
    for key, value in secondary_answers.items():
        secondary_details.append({value['text']: value['value']})

    summarized_response = {
        "ID": response_id,
        "Product Name": product_name,
        "Star Rating": star_rating,
        "Title": title,
        "Comment Feedback": comment_feedback,
        "Secondary Answers": secondary_details
    }

    return summarized_response

def fetch_process_and_store_data(start_page_number):
    page_number = start_page_number
    batch_number = 1

    while True:
        url = f"{base_url}?page%5Bnumber%5D={page_number}&page%5Bsize%5D=100"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            summarized_responses = []

            for survey_response in data['data']:
                summarized_response = process_and_summarize_response(survey_response)
                summarized_responses.append(summarized_response)

            output_file_path = f'summarized_responses_batch_{batch_number}.json'
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(summarized_responses, output_file, ensure_ascii=False, indent=4)

            print(f"Batch {batch_number} processed. Summarized responses saved to '{output_file_path}'.")

            batch_number += 1

            if 'next' in data['links']:
                page_number += 1
            else:
                print("No more 'next' link found. Pagination complete.")
                break
        else:
            print(f"Error occurred for page {page_number}: Status Code {response.status_code}")
            break

start_page_number = 1
fetch_process_and_store_data(start_page_number)
