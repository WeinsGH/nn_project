import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models import resnet50
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from prediction import As_File, show_file, ByURL, check_url


# Словарь сопоставления числовых предсказаний с классами
class_mapping = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake"
}

# Функция для классификации изображения
def classify_image(image, model):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_class = torch.max(output.data, 1)
    return predicted_class.item(), probabilities[predicted_class].item()


def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Home", "Preprocessing", "Parsing"])

    if selected_page == "Home":
        page_home()
    elif selected_page == "Preprocessing":
        page_preprocessing()
    elif selected_page == "Parsing":
        page_parsing()
def page_home():
    st.title("WhatIsItator for birds")

    col1, col2 = st.columns(2)

    with col1:
        selected1 = st.radio('Select model', ['resnet18', 'resnet50'])
        if selected1 == 'resnet18':
            model = resnet18(pretrained=False)
            restype = 'renet18'
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(torch.load('model/model.pth', map_location='cpu'))
        else:
            model = resnet50(pretrained=False)
            restype = 'renet50'
            model.fc = nn.Linear(2048, 10)
            model.load_state_dict(torch.load('model/model2.pth', map_location='cpu'))

    with col2:
        selected2 = st.radio('Select download', ['as file', 'by URL'])

    if selected2 == 'as file':
        uploaded_image = st.file_uploader("Choose an image...", type="jpg")
        model.eval()
        if uploaded_image is not None:
            st.write(As_File(uploaded_image, model, class_mapping))
            st.image(show_file(uploaded_image))
        else:
            pass
    else:
        url = st.text_input("Enter image URL:")
        answer = ByURL(url, model, class_mapping)
        st.write(answer)
        st.image(url, caption='Your picture', use_column_width=True)
        if ByURL(url, model, class_mapping) is not None:
            col3, col4 = st.columns(2)
            if col3.button("Согласен", key="agree_button"):
                st.success("Спасибо отклик!")
                feedback = 1
                if check_url(url, restype):
                    with open('data.txt', 'a') as file:
                        file.write(f"{url},{answer.split(' ')[2]},{answer.split(' ')[5]},{restype} {feedback}\n")
            if col4.button("Не согласен", key="disagree_button"):
                feedback = -1
                st.error("Спасибо за отклик!")
                if check_url(url, restype):
                    with open('data.txt', 'a') as file:
                        file.write(f"{url},{answer.split(' ')[2]},{answer.split(' ')[5]},{restype},{feedback}\n")




def page_preprocessing():
    st.title('Preprocessing')


def page_parsing():
    st.title('Parsing')
    if st.button('Show history'):
        df = pd.read_csv('data.txt', names=['url', 'Class', 'Confidence', 'Model', 'Feedback'])
        st.table(df)

if __name__ == "__main__":
    main()