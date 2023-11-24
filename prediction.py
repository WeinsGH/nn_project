from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models import resnet50
from torch.autograd import Variable
import torch.nn as nn
import requests
from io import BytesIO
import pandas as pd

def preprocess_image(image):
    # Преобразование изображения в формат RGB
    image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def As_File(uploaded_image, model, decode):
    model.eval()
    image = Image.open(uploaded_image).convert("RGB")
    image_tensor = preprocess_image(image).to('cpu')
    with torch.no_grad():
        outputs = model(image_tensor)

    # Преобразование предсказаний в вероятности с использованием Softmax
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Получение индекса предсказанного класса
    _, predicted_class = torch.max(outputs.data, 1)
    predicted_class_index = int(predicted_class.item())
    res = f'Predicted Class: {decode.get(predicted_class_index, f"Class {predicted_class_index}")} с уверенностью: {round((probabilities.max() * 100).item(), 2)} %'
    return res


def ByURL(url, model, decode):
    model.eval()
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    image_tensor = preprocess_image(img).to('cpu')
    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Получение индекса предсказанного класса
    _, predicted_class = torch.max(outputs.data, 1)
    predicted_class_index = int(predicted_class.item())
    res = f'Predicted Class: {decode.get(predicted_class_index, f"Class {predicted_class_index}")} с уверенностью: {round((probabilities.max() * 100).item(), 2)} %'
    return res
def show_file(img):
    return(img)

def check_url(url, restype):
    df = pd.read_csv('data.txt', names=['url', 'Class', 'Confidence', 'Model', 'Feedback'])
    if (url in df['url'].values) and (restype in df[df['url']==url]['Model'].values):
        return False
    else:
        return True


def parcing(model):
    pass