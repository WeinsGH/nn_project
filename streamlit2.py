import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

# Загрузка предварительно обученной модели
model = resnet18(pretrained=False)
# Замените 'path/to/your/model.pth' на путь к вашим весам модели
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('model/model.pth', map_location='cpu'))
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

st.write(print(model.fc.weight))
model.eval()


# Функция для предобработки изображения перед подачей в модель
def preprocess_image(image):
    # Преобразование изображения в формат RGB
    image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Функция для классификации изображения
def classify_image(image, model):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(Variable(image_tensor))
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_class = torch.max(output.data, 1)
    return predicted_class.item(), probabilities[predicted_class].item()

# Основная часть Streamlit-приложения
def main():
    st.title("Image Classification App")
    uploaded_image = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        
        # # Классификация изображения
        # predicted_class, confidence = classify_image(image, model)

        # st.write(f"Prediction: Class {predicted_class}")
        # st.write(f"Confidence: {confidence:.2%}")

        # Предобработка изображения
        image_tensor = preprocess_image(image).to('cpu')
        ## Предсказание с помощью модели
        with torch.no_grad():
            outputs = model(image_tensor)

        # Преобразование предсказаний в вероятности с использованием сигмоиды
        predicted_probs = torch.sigmoid(outputs)

        # Получение индекса предсказанного класса
        _, predicted_classes = torch.max(predicted_probs, 1)
        predicted_class_index = int(predicted_classes[0].item())

        # Отображение изображения и подпись предсказанного класса
        image = Image.open(uploaded_image)
        plt.imshow(image)
        predicted_class_name = class_mapping.get(predicted_class_index, f'Class {predicted_class_index}')
        plt.title(f'Predicted Class: {predicted_class_name}')
        plt.show()

        
if __name__ == "__main__":
    main()
