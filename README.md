# LLM unisup naming Classifier

This repository contains a Streamlit web app that deploys a pre-trained LLM (Large Language Model) unisup naming classifier for inference. The app allows users to classify text as either "Proper Naming Notfcn" or "Wrong Naming Notificn" using two input options:

1. Text Input: Users can enter a single text string to classify.
2. File Upload: Users can upload an Excel file containing multiple texts to classify.

The app uses a pre-trained LLM model to perform the classification and displays the predicted labels for each input text.

## Installation

1. Clone the repository:
git clone https://github.com/your-username/llm-spam-classifier.git
2. Install the required dependencies:
pip install -r requirements.txt

## Usage

1. Navigate to the project directory:
cd llm-spam-classifier

2. Run the Streamlit app:
streamlit run app.py

3. Access the app in your web browser and use the provided input options to classify texts.

## Model

The app uses a pre-trained LLM model for spam classification. The model checkpoint is stored in the `review_classifier.pth` file.

## License

This project is licensed under the [MIT License](LICENSE).
