from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .serializers import ImageUploadSerializer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Load once globally
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../saved_model/Brain_tumor_mobilenetV3.keras')
model = load_model(MODEL_PATH)

# Your class labels
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            image = Image.open(image_file).convert("RGB")
            image = image.resize((224, 224))  # Match model input size

            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_label = class_labels[predicted_class_idx]  # Use same class_labels dict
            confidence = float(np.max(prediction))

            return Response({
                'predicted_label': predicted_label,
                'confidence': round(confidence * 100, 2),
                'class_probabilities': {
                    class_labels[i]: round(float(prob) * 100, 2)
                    for i, prob in enumerate(prediction[0])
                }
            })
        return Response(serializer.errors, status=400)
