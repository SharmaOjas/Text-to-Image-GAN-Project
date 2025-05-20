import torch
from transformers import BertTokenizer, BertModel
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms, models
from PIL import Image
import streamlit as st

st.title("Integrated Text-to-Image & Classification App")

task = st.selectbox("Choose Task", [
    "Task 1: Text Encoding (BERT)", 
    "Task 2: Text Embeddings (CLIP)", 
    "Task 3: Image Classification (Color)"
])

if task == "Task 1: Text Encoding (BERT)":
    st.header("BERT Text Tokenization and Encoding")

    bert_model_path = "models/task1_bert"
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = BertModel.from_pretrained(bert_model_path, device_map="auto")
    model.eval()

    user_text = st.text_area("Enter text to encode", "Type something here...")

    if user_text:
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        st.subheader("Token IDs")
        st.write(inputs['input_ids'].cpu().numpy())

        st.subheader("BERT Embeddings (CLS token)")
        st.write(embeddings)


elif task == "Task 2: Text Embeddings (CLIP)":
    st.header("CLIP Text Embeddings for Text-to-Image")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load both tokenizer and model from the same pretrained source
    clip_model_path = "openai/clip-vit-base-patch32"
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    clip_model = CLIPTextModel.from_pretrained(clip_model_path)
    
    # If using local weights, ensure they match the tokenizer
    # clip_model_path = "models/task2_clip"
    # clip_model = CLIPTextModel.from_pretrained(clip_model_path)

    clip_model.to(device)
    clip_model.eval()

    descriptions = st.text_area("Enter text descriptions (comma separated)", "a photo of a cat, a photo of a dog")

    if descriptions:
        text_list = [desc.strip() for desc in descriptions.split(",")]

        # Tokenize inputs
        inputs = clip_tokenizer(
            text_list,
            padding="max_length",
            truncation=True,
            max_length=clip_tokenizer.model_max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = clip_model(**inputs)
                # Use pooler_output for a single embedding per input (2D: batch_size, hidden_size)
                embeddings = outputs.pooler_output.cpu().numpy()
                # Alternatively, mean-pool the last_hidden_state:
                # embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            st.subheader("Text Embeddings Shape")
            st.write(embeddings.shape)

            st.subheader("Embeddings")
            st.write(embeddings)
        except Exception as e:
            st.error(f"Error processing embeddings: {str(e)}")

elif task == "Task 3: Image Classification (Color)":
    st.header("Color Classification of Uploaded Images")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    num_classes = 5  # Adjust based on your dataset
    model.fc = torch.nn.Linear(num_features, num_classes)

    model_path = "models/task3/color_classifier_full.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        class_names = ["Red", "Blue", "Green", "Yellow", "Black"]  # Update as per your dataset
        st.write(f"Predicted color class: **{class_names[pred]}**")
