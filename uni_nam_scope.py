import streamlit as st
import pandas as pd
import torch
from transformers import GPT2Tokenizer
from io import BytesIO
import requests

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the classification function
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    # Truncate sequences if they are too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "Proper Naming Notfcn" if predicted_label == 1 else "Wrong Naming Notificn"

# Streamlit app
def main():
    st.title("Unisup Naming Classifier")

    # Load the trained model from GitHub
    model_url = "https://github.com/achilela/unisup_naming_classifier/main/review_classifier.pth"
    model_state_dict = torch.hub.load_state_dict_from_url(model_url)
    model = ...  # Instantiate your model architecture
    model.load_state_dict(model_state_dict)
    model.eval()

    # Sidebar options
    input_option = st.sidebar.selectbox("Select Input Option", ["Text Input", "File Upload"])

    if input_option == "Text Input":
        # Text input for single text classification
        text_input = st.text_input("Enter the text to classify")
        if st.button("Classify"):
            if text_input:
                # Classify the text
                predicted_label = classify_review(text_input, model, tokenizer, device, max_length=train_dataset.max_length)
                st.success(f"Predicted Label: {predicted_label}")
            else:
                st.warning("Please enter some text to classify.")

    elif input_option == "File Upload":
        # File uploader for classifying texts from an Excel file
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
        if uploaded_file is not None:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            text_column = st.selectbox("Select the column containing the texts", df.columns)

            # Classify the texts
            predicted_labels = []
            for text in df[text_column]:
                predicted_label = classify_review(text, model, tokenizer, device, max_length=train_dataset.max_length)
                predicted_labels.append(predicted_label)

            # Add the predicted labels to the DataFrame
            df["Predicted Label"] = predicted_labels

            # Display the results
            st.write(df)

if __name__ == "__main__":
    main()
