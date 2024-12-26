# # from django.shortcuts import render
# # from django.views.decorators.csrf import csrf_exempt
# # from django.http import JsonResponse
# # import numpy as np
# # import urllib.request  # For Python 3
# # import json
# # import cv2
# # import os

# # # Define the path to the face detector
# # FACE_DETECTOR_PATH = os.path.join(
# #     os.path.abspath(os.path.dirname(__file__)), "cascades", "haarcascade_frontalface_default.xml"
# # )

# # @csrf_exempt
# # def detect(request):
# #     """
# #     Handle face detection requests. Accepts POST requests with an image or image URL.
# #     """
# #     # Initialize the data dictionary to be returned by the request
# #     data = {"success": False}

# #     # Check if this is a POST request
# #     if request.method == "POST":
# #         # Check if an image was uploaded directly
# #         if request.FILES.get("image", None) is not None:
# #             # Grab the uploaded image
# #             image = _grab_image(stream=request.FILES["image"])
# #         else:
# #             # Otherwise, assume a URL was passed in
# #             url = request.POST.get("url", None)

# #             # If no URL is provided, return an error
# #             if url is None:
# #                 data["error"] = "No URL provided."
# #                 return JsonResponse(data)

# #             # Load the image from the provided URL
# #             image = _grab_image(url=url)

# #         # Convert the image to grayscale
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #         # Load the face cascade detector and detect faces in the image
# #         detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
# #         rects = detector.detectMultiScale(
# #             image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
# #         )

# #         # Construct a list of bounding boxes for detected faces
# #         rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

# #         # Update the data dictionary with the detected faces
# #         data.update({"num_faces": len(rects), "faces": rects, "success": True})

# #     # Return the response as JSON
# #     return JsonResponse(data)

# # def _grab_image(path=None, stream=None, url=None):
# #     """
# #     Load an image from a file path, URL, or uploaded file stream.
# #     """
# #     # If the path is provided, load the image from disk
# #     if path is not None:
# #         image = cv2.imread(path)
# #     else:
# #         # If a URL is provided, download the image
# #         if url is not None:
# #             resp = urllib.request.urlopen(url)
# #             data = resp.read()
# #         # If a stream is provided, read the uploaded image
# #         elif stream is not None:
# #             data = stream.read()

# #         # Convert the image data to a NumPy array and then to OpenCV format
# #         image = np.asarray(bytearray(data), dtype="uint8")
# #         image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# #     # Return the loaded image
# #     return image

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib.request  # For Python 3
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from django.http import JsonResponse
from .models import model
# from face_detector.load_model import load_pretrained_model


# Загрузка модели TensorFlow при запуске сервера
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "/Users/pavloiermakov/Desktop/mysite/face_detector/models/models.h5")
model = load_model(MODEL_PATH)

# Define the path to the face detector
FACE_DETECTOR_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "cascades", "haarcascade_frontalface_default.xml"
)

@csrf_exempt
def detect(request):
    """
    Handle face detection requests. Accepts POST requests with an image or image URL.
    """
    # Initialize the data dictionary to be returned by the request
    data = {"success": False}

    if request.method == "POST":
        try:
            # Check if an image was uploaded directly
            if request.FILES.get("image", None) is not None:
                image = _grab_image(stream=request.FILES["image"])
            else:
                # Assume a URL was passed in
                url = request.POST.get("url", None)
                if not url:
                    data["error"] = "No URL provided."
                    return JsonResponse(data)
                image = _grab_image(url=url)

            # Convert the image to grayscale for face detection
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Load the face cascade detector and detect faces in the image
            detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
            rects = detector.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Create bounding boxes for detected faces
            rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

            # Draw rectangles on the original image
            for (x1, y1, x2, y2) in rects:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save the processed image to the media directory
            processed_image_path = os.path.join("media", "processed_image.jpg")
            cv2.imwrite(processed_image_path, image)

            # Update the response data
            data.update({
                "success": True,
                "num_faces": len(rects),
                "faces": rects,
                "image_url": "/" + processed_image_path  # URL to access the processed image
            })

        except Exception as e:
            data["error"] = f"An error occurred: {str(e)}"

    # Return the response as JSON
    return JsonResponse(data)

def _grab_image(path=None, stream=None, url=None):
    """
    Load an image from a file path, URL, or uploaded file stream.
    """
    try:
        if path:
            image = cv2.imread(path)
        elif url:
            resp = urllib.request.urlopen(url)
            data = resp.read()
            image = np.asarray(bytearray(data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        elif stream:
            data = stream.read()
            image = np.asarray(bytearray(data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            raise ValueError("No valid source for the image provided.")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def result(request):
    """
    Render the results page with dynamically detected face data.
    """
    # Retrieve the number of faces and processed image URL from the session
    num_faces = request.session.get("num_faces", 0)  # Default to 0 if not set
    processed_image_url = "/media/processed_image.jpg"

    return render(request, 'face/results.html', {
        "image_url": processed_image_url,
    })

def index(request):
    """
    Render the index page where users can upload an image or provide a URL.
    """
    return render(request, 'face/index.html', {})

# def classify_image(request):
#     """
#     Handle image classification and render results in classify.html.
#     """
#     if request.method == "POST" and request.FILES.get("image", None):
#         try:
#             # Get the uploaded image
#             file = request.FILES["image"]
#             image = np.asarray(bytearray(file.read()), dtype="uint8")
#             image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#             # Preprocess the image
#             image = cv2.resize(image, (224, 224))  # Resize to match the model's input size
#             image = np.expand_dims(image, axis=0)
#             image = preprocess_input(image)

#             # Get predictions from the model
#             predictions = model.predict(image)
#             decoded_predictions = decode_predictions(predictions, top=3)[0]

#             # Prepare the results for rendering
#             results = [{"label": label, "description": description, "score": float(score)}
#                        for (_, label, description, score) in decoded_predictions]

#             return render(request, 'face/classify.html', {"predictions": results})
#         except Exception as e:
#             return render(request, 'face/classify.html', {"error": str(e)})

#     return render(request, 'face/classify.html', {"error": "Загрузите изображение для классификации."})
def classify_image(request):
    """
    Handle image classification and render results in classify.html.
    """
    print("Entering classify_image function")
    if request.method == "POST" and request.FILES.get("image", None):
        print("POST request detected with an uploaded image.")
        try:
            # Get the uploaded image
            file = request.FILES["image"]
            print(f"Received file: {file.name}")
            
            image = np.asarray(bytearray(file.read()), dtype="uint8")
            print("Image successfully converted to NumPy array.")
            
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            print("Image decoded using OpenCV.")

            # Preprocess the image
            image = cv2.resize(image, (224, 224))  # Resize to match the model's input size
            print("Image resized to 224x224.")
            
            image = np.expand_dims(image, axis=0)
            print("Image expanded to match model input shape.")
            
            image = preprocess_input(image)
            print("Image preprocessed using preprocess_input.")

            # Get predictions from the model
            predictions = model.predict(image)
            print(f"Raw model predictions: {predictions}")
            
            decoded_predictions = decode_predictions(predictions, top=3)[0]
            print(f"Decoded predictions: {decoded_predictions}")

            # Prepare the results for rendering
            results = [{"label": label, "description": label, "score": float(score)}
                       for (_, label, score) in decoded_predictions]
            print(f"Formatted results: {results}")

            return render(request, 'face/classify.html', {"predictions": results})
        except Exception as e:
            print(f"An error occurred during classification: {e}")
            return render(request, 'face/classify.html', {"error": str(e)})

    print("No valid POST request or image uploaded.")
    return render(request, 'face/classify.html', {"error": "Загрузите изображение для классификации."})
