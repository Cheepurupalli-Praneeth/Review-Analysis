# import requests
# import json

# # Base URL of the API
# base_url = "https://data.g2.com/api/v1/survey-responses"

# # Authorization token and headers
# headers = {
#     "Authorization": "Token token=c1725e36612f450e1e0ae0b656a65f691d3d98d41308ba54059b28f32ecb9080",
#     "Content-Type": "application/vnd.api+json"
# }

# # Function to fetch and store data for a given page number
# def fetch_and_store_data(page_number):
#     # Construct the URL with the specific page number
#     url = f"{base_url}?page%5Bnumber%5D={page_number}&page%5Bsize%5D=100"

#     # Send GET request
#     response = requests.get(url, headers=headers)

#     # Check if the request was successful
#     if response.status_code == 200:
#         # Parse the JSON response
#         data = response.json()

#         # Store the data in a JSON file named with the page number
#         filename = f"survey_responses_page_{page_number}.json"
#         with open(filename, 'w') as json_file:
#             json.dump(data, json_file)

#         print(f"Data for page {page_number} has been extracted and stored in '{filename}'.")
#     else:
#         print(f"Error occurred for page {page_number}: Status Code {response.status_code}")

# # Define the total number of pages
# total_pages = 8

# # Loop through each page and fetch data
# for page_number in range(1, total_pages + 1):
#     fetch_and_store_data(page_number)
import requests
import json

# Base URL of the API
base_url = "https://data.g2.com/api/v1/survey-responses"

# Authorization token and headers
headers = {
    "Authorization": "Token token=c1725e36612f450e1e0ae0b656a65f691d3d98d41308ba54059b28f32ecb9080",
    "Content-Type": "application/vnd.api+json"
}

# Function to fetch and store data starting from a given page number
def fetch_and_store_data(start_page_number):
    page_number = start_page_number

    while True:
        # Construct the URL for the current page number
        url = f"{base_url}?page%5Bnumber%5D={page_number}&page%5Bsize%5D=100"

        # Send GET request
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Store the data in a JSON file named with the page number
            filename = f"survey_responses_page_{page_number}.json"
            with open(filename, 'w') as json_file:
                json.dump(data, json_file)

            print(f"Data for page {page_number} has been extracted and stored in '{filename}'.")

            # Check if there's a next page link
            if 'next' in data['links']:
                page_number += 1  # Increment page number for the next request
            else:
                print("No more 'next' link found. Pagination complete.")
                break
        else:
            print(f"Error occurred for page {page_number}: Status Code {response.status_code}")
            break

# Define the starting page number (e.g., 1 for the first page)
start_page_number = 1

# Start fetching data from the specified page number
fetch_and_store_data(start_page_number)

