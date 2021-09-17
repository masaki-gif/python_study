from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps 
#from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['input_file'].stream
    im = Image.open(data)

    model = load_model('mnist_model_weight.h5') 

    img = im.resize((28,28))
    img = img.convert(mode='L')
    img = ImageOps.invert(img) 
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')/255

    result = model.predict_classes(img)
    result = result[0]

    return render_template('result.html',result_output=result)

if __name__ == '__main__':
    app.run(debug=True)