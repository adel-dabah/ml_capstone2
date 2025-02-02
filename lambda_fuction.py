#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
import numpy as np 
import requests
from io import BytesIO
from PIL import Image
print("Hello")
# Load the TFLite model and allocate tensors.

interpreter =tflite.Interpreter(model_path="mri.model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_index=interpreter.get_input_details()[0]['index']
output_index=interpreter.get_output_details()[0]['index']

# Prepare the input data
#url = "https://github.com/adel-dabah/ml_capstone2/blob/main/dataset/Testing/pituitary/Te-piTr_0003.jpg?raw=true"
def predict(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((300, 300))
    x = np.array(img)
    X=np.array([x])

    #X=preprocess_input(X)
    interpreter.set_tensor(input_index, X.astype(np.float32))

    #run the inference
    interpreter.invoke()

    #get the output tensor
    pred_x=interpreter.get_tensor(output_index)

    float_pred=pred_x[0].tolist()
    class_indices={'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

    pred=dict(zip(list(class_indices.keys()), float_pred))


    #print(pred)
    return pred

#print(predict(url))

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    return (pred)



