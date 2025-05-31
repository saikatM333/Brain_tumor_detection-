from django.shortcuts import render

# Create your views here.
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser
# from .serializers import ImageUploadSerializer
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from PIL import Image
# import numpy as np
# import os

# # Load once globally
# MODEL_PATH = os.path.join(os.path.dirname(__file__), '../saved_model/Brain_tumor_mobileNetV2.keras')
# model = load_model(MODEL_PATH)

# # Your class labels
# CLASS_LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# class PredictView(APIView):
#     parser_classes = [MultiPartParser]

#     def post(self, request):
#         serializer = ImageUploadSerializer(data=request.data)
#         if serializer.is_valid():
#             image_file = serializer.validated_data['image']
#             image = Image.open(image_file).convert("RGB")
#             image = image.resize((160, 160))  # Your model input size
#             img_array = img_to_array(image) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)

#             prediction = model.predict(img_array)
#             predicted_class_idx = np.argmax(prediction, axis=1)[0]
#             predicted_label = CLASS_LABELS[predicted_class_idx]
#             confidence = float(np.max(prediction))

#             return Response({
#                 'predicted_label': predicted_label,
#                 'confidence': round(confidence * 100, 2),
#                 'class_probabilities': {CLASS_LABELS[i]: round(float(prob)*100, 2) for i, prob in enumerate(prediction[0])}
#             })
#         return Response(serializer.errors, status=400)

# views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from .serializers import ImageUploadSerializer

# Load your trained model (make sure the path is correct)
model = load_model('best_mobilenet_model.h5')

# Match the class labels to the order used during training
CLASS_LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary_tumor']  # Example — adjust to your dataset folders

IMG_SIZE = 224

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']

            # Preprocess image
            image = Image.open(image_file).convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(image) / 255.0  # Rescale just like training
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_label = CLASS_LABELS[predicted_class_idx]
            confidence = float(np.max(prediction))

            return Response({
                'predicted_label': predicted_label,
                'confidence': round(confidence * 100, 2),
                'class_probabilities': {
                    CLASS_LABELS[i]: round(float(prob) * 100, 2) for i, prob in enumerate(prediction[0])
                }
            })

        return Response(serializer.errors, status=400)
