from flask import Flask, render_template, request, redirect, url_for
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Function to process image and return the detected class
def detect_image(image_path):
    # # Implement your image detection logic here
    # # This function should return the detected class
    # # For simplicity, let's return a random class
    # classes = ['Class 1', 'Class 2', 'Class 3']
    # import random
    # return random.choice(classes)
    model = load_model('vgg-rps-final.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    y_classes = prediction.argmax(axis=-1)
    accu=prediction[0][y_classes]*100
    # print(accu)
    
    if y_classes==0:
        print("CoVid with accuracy: ",accu)
        # return "Covid with accuracy",accu
        detected_class = "Covid "
        accu=accu
        
    elif y_classes==1:
        print("Normal with accuracy: ",accu)
        # return "Normal with accuracy "
        detected_class = "Normal"
        accu=accu
        
    elif y_classes==2:
        print("Pnuemonia with accuracy: ",accu)
        # return "Pnuemonia with accuracy"
        detected_class = "Pneumonia"
        accu=accu

    return detected_class, accu
        



@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Save the uploaded file
        file = request.files['file']
        if file:
            filename = 'uploaded_image.jpg'
            file.save(os.path.join('static', filename))
            return redirect(url_for('result'))
    return render_template('upload.html')

@app.route('/result')
def result():
    image_path = os.path.join('static', 'uploaded_image.jpg')
    detected_class,accuracy = detect_image(image_path)
    # accu=detect_image(accu)
    return render_template('result.html', image_path=image_path, detected_class=detected_class,accuracy=accuracy)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            # Redirect to the next page upon successful login
            return redirect(url_for('upload'))
        else:
            return 'Invalid username or password. Please try again.'
    return render_template('login.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)
