# Survey Response Analysis

The **Survey Response Analysis** project is designed to automate the process of fetching, processing, and analyzing survey responses from a data source using an API. The system extracts valuable insights from customer feedback to identify specific feature sets that customers are looking for in various products or services.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

In today's data-driven world, businesses rely heavily on customer feedback to improve their products and services. Analyzing survey responses manually can be time-consuming and error-prone. This project aims to streamline the analysis process by leveraging automation and natural language processing techniques.

## Features

- **Survey Response Fetching**: Retrieve survey responses from a specified API endpoint.
  
- **Data Processing and Summarization**: Extract key information such as product names, star ratings, feedback comments, and secondary answers.
  
- **Feature Generation**: Use a pretrained language model to generate feature sets based on customer feedback to identify specific aspects or attributes that customers prioritize.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/survey-response-analysis.git
   cd survey-response-analysis 

2. Install dependencies:
   ```bash
   pip install -r requirements.txt 


### Configuration
1. Obtain an access token for the API.
2. Create a config.json file in the project directory with the following structure:
   ```json
   {
    "API_KEY": "your-api-token-here"
   }

Replace "your-api-token-here" with your actual API token.

### Usage
1. Open main.py and ensure the BASE_URL constant is set to your API endpoint.
2. Execute the main script:
   ```bash
   python main.py

The script will fetch survey responses, process them, and generate feature sets based on customer feedback using the provided access token.

**Code Structure**
- **`main.py`**: Contains the main script to fetch survey responses and process them.
- **`config.json`**: Stores the API access token (not included in version control).
- **`requirements.txt`**: Lists the required Python packages.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request.
