import streamlit as st
import torch
from transformers import GPT2Tokenizer
import pandas as pd

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

# Load the trained model from the local directory
model_path = "clv__classifier"
model = torch.load(model_path)
model.eval()

# Set the device to run the model on (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit app
def main():
    st.title("Text Classification App")

    # Input options
    input_option = st.radio("Select input option", ("Single Text Query", "Upload Table"))

    if input_option == "Single Text Query":
        # Single text query input
        text_query = st.text_input("Enter text query")
        if st.button("Classify"):
            if text_query:
                # Classify the text query
                predicted_label = classify_review(text_query, model, tokenizer, device, max_length=train_dataset.max_length)
                st.write("Predicted Label:")
                st.write(predicted_label)
            else:
                st.warning("Please enter a text query.")

    elif input_option == "Upload Table":
        # Table upload
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            # Read the uploaded file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Select the text column
            text_column = st.selectbox("Select the text column", df.columns)

            # Classify the texts in the selected column
            predicted_labels = []
            for text in df[text_column]:
                predicted_label = classify_review(text, model, tokenizer, device, max_length=train_dataset.max_length)
                predicted_labels.append(predicted_label)

            # Add the predicted labels to the DataFrame
            df["Predicted Label"] = predicted_labels

            # Display the DataFrame with predicted labels
            st.write(df)

if __name__ == "__main__":
    main()
