from io import BytesIO
from urllib import request
from PIL import Image
import onnxruntime as ort
import numpy as np
import json

session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def lambda_handler(event, context):
    url = event["url"]

    img = download_image(url)
    img = prepare_image(img)

    x = np.array(img).astype("float32") / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std

    x = np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)

    pred = session.run([output_name], {input_name: x})[0]
    prob = float(pred[0][0])

    return {
        "probability": prob
    }
