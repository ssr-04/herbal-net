import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms, models  # datsets  , transforms
import torch.nn as nn
import torch.nn.functional as F


label_names = ['Alpinia Galanga (Rasna)',
 'Amaranthus Viridis (Arive-Dantu)',
 'Artocarpus Heterophyllus (Jackfruit)',
 'Azadirachta Indica (Neem)',
 'Basella Alba (Basale)',
 'Brassica Juncea (Indian Mustard)',
 'Carissa Carandas (Karanda)',
 'Citrus Limon (Lemon)',
 'Ficus Auriculata (Roxburgh fig)',
 'Ficus Religiosa (Peepal Tree)',
 'Hibiscus Rosa-sinensis',
 'Jasminum (Jasmine)',
 'Mangifera Indica (Mango)',
 'Mentha (Mint)',
 'Moringa Oleifera (Drumstick)',
 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
 'Murraya Koenigii (Curry)',
 'Nerium Oleander (Oleander)',
 'Nyctanthes Arbor-tristis (Parijata)',
 'Ocimum Tenuiflorum (Tulsi)',
 'Piper Betle (Betel)',
 'Plectranthus Amboinicus (Mexican Mint)',
 'Pongamia Pinnata (Indian Beech)',
 'Psidium Guajava (Guava)',
 'Punica Granatum (Pomegranate)',
 'Santalum Album (Sandalwood)',
 'Syzygium Cumini (Jamun)',
 'Syzygium Jambos (Rose Apple)',
 'Tabernaemontana Divaricata (Crape Jasmine)',
 'Trigonella Foenum-graecum (Fenugreek)']


# Load the trained model
model = models.resnet50(pretrained=False)  # Create the model architecture
num_classes = 30  # Replace with the number of classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the last classification layer
model.load_state_dict(torch.load("./herbal-net-v1.pth",map_location=torch.device('cpu')))
model.eval()

# Define the image transformation for the real-world image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_model(version,path):
    # Load the real-world image
    image_path = path  # Replace with the path to your image
    image = Image.open(image_path)

    # Apply the transformation to preprocess the image
    input_image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Make predictions
    with torch.no_grad():
        output = model(input_image)

    # Apply softmax to get class probabilities
    probabilities = torch.softmax(output, dim=1)[0]  # Get the probabilities for the first (and only) image in the batch

    # Get the predicted class label
    _, predicted = torch.max(output, 1)
    predicted_class = label_names[predicted.item()]

    # Display the predicted class and its associated probability
    label = predicted_class
    confidence = probabilities[predicted.item()].item() * 100

    return([label,confidence])

def get_details(name):
    herbal_plants = {
    'Alpinia Galanga (Rasna)': 'A medicinal herb used for its anti-inflammatory properties, particularly in traditional Ayurvedic medicine.',
    'Amaranthus Viridis (Arive-Dantu)': 'A leafy vegetable rich in vitamins and minerals, known for its antioxidant and anti-inflammatory effects.',
    'Artocarpus Heterophyllus (Jackfruit)': 'A tropical fruit with edible seeds and flesh, providing essential nutrients and fiber.',
    'Azadirachta Indica (Neem)': 'A versatile plant used for its antibacterial, antifungal, and antiviral properties in various herbal remedies.',
    'Basella Alba (Basale)': 'A leafy green vegetable packed with vitamins and minerals, often included in traditional diets for its nutritional value.',
    'Brassica Juncea (Indian Mustard)': 'The seeds and leaves of this plant are used for their culinary and medicinal properties, aiding digestion and providing essential nutrients.',
    'Carissa Carandas (Karanda)': 'Known for its medicinal fruit, it\'s used in traditional medicine for its antioxidant and anti-inflammatory benefits.',
    'Citrus Limon (Lemon)': 'Rich in vitamin C, lemons are used for their detoxifying and immune-boosting properties.',
    'Ficus Auriculata (Roxburgh fig)': 'A fig species with edible fruits, often consumed for their nutritional value.',
    'Ficus Religiosa (Peepal Tree)': 'Considered sacred in Hinduism, it\'s used in traditional medicine for its various health benefits.',
    'Hibiscus Rosa-sinensis': 'The flowers are used to make herbal teas known for their potential in lowering blood pressure and promoting hair health.',
    'Jasminum (Jasmine)': 'Jasmine flowers are used in aromatherapy for their calming and soothing effects.',
    'Mangifera Indica (Mango)': 'Delicious tropical fruit rich in vitamins, minerals, and antioxidants.',
    'Mentha (Mint)': 'Known for its soothing properties, mint is used in teas and remedies for digestive and respiratory issues.',
    'Moringa Oleifera (Drumstick)': 'A highly nutritious plant with leaves and pods used for their vitamins and minerals.',
    'Muntingia Calabura (Jamaica Cherry-Gasagase)': 'The fruit is rich in vitamin C and antioxidants, used for its potential health benefits.',
    'Murraya Koenigii (Curry)': 'Curry leaves are used to flavor dishes and are believed to have medicinal properties for hair and skin.',
    'Nerium Oleander (Oleander)': 'Contains toxic compounds and should be used with caution; traditionally used in some cultures for medicinal purposes.',
    'Nyctanthes Arbor-tristis (Parijata)': 'The leaves have medicinal properties and are used in Ayurvedic treatments for various ailments.',
    'Ocimum Tenuiflorum (Tulsi)': 'Holy basil, used in Ayurvedic medicine for its adaptogenic and immune-boosting properties.',
    'Piper Betle (Betel)': 'Betel leaves are chewed with areca nut for their stimulating and medicinal effects.',
    'Plectranthus Amboinicus (Mexican Mint)': 'Used in traditional medicine for its potential respiratory and digestive benefits.',
    'Pongamia Pinnata (Indian Beech)': 'The oil from its seeds is used for its medicinal properties in traditional remedies.',
    'Psidium Guajava (Guava)': 'A fruit rich in vitamin C and dietary fiber, known for its digestive and immune-boosting properties.',
    'Punica Granatum (Pomegranate)': 'The fruit is loaded with antioxidants and is known for its potential heart and skin health benefits.',
    'Santalum Album (Sandalwood)': 'Sandalwood oil is used in aromatherapy and skincare for its calming and soothing effects.',
    'Syzygium Cumini (Jamun)': 'The fruit is known for its potential in managing diabetes due to its low glycemic index.',
    'Syzygium Jambos (Rose Apple)': 'The fruit is rich in vitamin C and used for its potential health benefits.',
    'Tabernaemontana Divaricata (Crape Jasmine)': 'The plant is used in traditional medicine for its potential antispasmodic and sedative effects.',
    'Trigonella Foenum-graecum (Fenugreek)': 'The seeds are used in cooking and herbal remedies, known for their potential in regulating blood sugar and improving digestion.'
    }

    return herbal_plants.get(name)
