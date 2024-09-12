from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")


import os
os.makedirs('/content/static', exist_ok=True)

import shutil
shutil.copy('/content/drive/My Drive/Web Images/bgi.jpg', '/content/static/background.jpeg')
shutil.copy('/content/drive/My Drive/Web Images/upimg.jpeg', '/content/static/upimg.jpeg')

from flask import Flask, request, render_template_string, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

model_path = '/content/drive/My Drive/models/finalmodel.keras'
try:
    model = load_model(model_path)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (200, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image


def predict_pneumonia(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    if prediction[0][0] >= 0.6:
        return f"X-Ray Shows Signs of Pneumonia"
    if prediction[0][0]>=0.5 and prediction[0][0]<0.6:
        return f"X-Ray Shows Possible Early Stages of Pneumonia"
    else:
        return f"X-Ray Shows No Signs of Pneumonia"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return "No file part"
            file = request.files['file']
            if file.filename == '':
                return "No selected file"
            if file:
                file_path = os.path.join('/content/uploads', file.filename)
                file.save(file_path)
                logging.info(f"File saved to: {file_path}")
                result = predict_pneumonia(file_path)
                return render_template_string(result_page, result=result, image_path='/uploads/' + file.filename)
        return render_template_string(upload_page)
    except Exception as e:
        logging.error(f"Error in upload_file: {e}")
        return f"An error occurred during file upload: {e}"
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory('/content/uploads', filename)
    except Exception as e:
        logging.error(f"Error in send_from_directory: {e}")
        return f"An error occurred while accessing the file: {e}"

upload_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Pneumonia Detector</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <style>
        body {font-family: "Times New Roman", Georgia, Serif;}
        h1, h2, h3, h4, h5, h6 {
            font-family: "Playfair Display";
            letter-spacing: 5px;
        }
        .custom-file-upload, input[type="submit"] {
            background-color: #888;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            display: inline-block;
        }
        .custom-file-upload:hover, input[type="submit"]:hover {
            background-color: #555;
        }
        .container {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .input-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="file"] {
            display: none;
        }
    </style>
</head>
<body>

<!-- Navbar (sit on top) -->
<div class="w3-top">
    <div class="w3-bar w3-white w3-padding w3-card" style="letter-spacing:4px;">
        <a href="#" class="w3-bar-item">Pneumonia Detector</a>
    </div>
</div>

<!-- Header -->
<header class="w3-display-container w3-content w3-wide" style="max-width:1600px;min-width:500px" id="home">
    <div class="w3-display-bottomleft w3-padding-large w3-opacity">
        </div>
</header>

<!-- Page content -->
<div class="w3-content" style="max-width:1100px">

    <!-- About Section -->
    <div class="w3-row w3-padding-32" id="about" style="margin-top: 50px;">
        <div class="w3-col m6 w3-padding-large w3-hide-small" style="margin-top:120px;margin-right-50px">
            <img src="/static/upimg.jpeg" class="w3-round w3-image w3-opacity-min" alt="Table Setting" width="600" height="750">
        </div>

        <div class="w3-col m6 w3-padding-large">
            <h1 class="w3-center">About Pneumonia</h1><br>
            <p class="w3-large">Pneumonia is a major global health concern, affecting millions of people each year and leading to significant morbidity and mortality.
             Traditional methods of pneumonia diagnosis, such as physical examination and chest X-rays, can be time-consuming, subjective, and prone to inaccuracies, particularly in areas with limited healthcare resources.
             Developing an AI-based pneumonia detection system could revolutionize the way this disease is identified and treated, leading to faster diagnoses, more accurate treatment, and improved patient outcomes.</p>
            <p class="w3-large">It's important to note that normal people can't just look at X-rays and tell whether it's pneumonic. Even experienced radiologists can sometimes struggle to accurately diagnose pneumonia from
            chest X-rays alone. This highlights the need for additional tools, such as AI-based systems, to aid in the diagnosis and treatment of this disease.</p>
        </div>
    </div>

    <!-- Upload Section -->
    <div class="w3-container w3-padding-64" id="upload">
        <div class="container">
            <h1>Upload X-Ray Image for Analysis</h1>
            <form method="POST" enctype="multipart/form-data">
                <div class="input-wrapper">
                    <label for="file-upload" class="custom-file-upload">
                        Choose File
                    </label>
                    <input id="file-upload" type="file" name="file">
                    <input type="submit" value="Upload">
                </div>
            </form>
        </div>
    </div>

 <hr>

    <!-- Contact Section -->
    <div class="w3-container w3-padding-64" id="contact">
        <h1>Contact</h1><br>
        <p>If you have any questions or need further information, feel free to contact us.</p>
        <p class="w3-text-blue-grey w3-large"><b></b></p>
        <form action="/action_page.php" target="_blank">
            <p><input class="w3-input w3-padding-16" type="text" placeholder="Name" required name="Name"></p>
            <p><input class="w3-input w3-padding-16" type="number" placeholder="How many people" required name="People"></p>
            <p><input class="w3-input w3-padding-16" type="datetime-local" placeholder="Date and time" required name="date" value="2020-11-16T20:00"></p>
            <p><input class="w3-input w3-padding-16" type="text" placeholder="Message / Special requirements" required name="Message"></p>
            <p><button class="w3-button w3-light-grey w3-section" type="submit">SEND MESSAGE</button></p>
        </form>
    </div>

<!-- End page content -->
</div>

<!-- Footer -->
<footer class="w3-center w3-light-grey w3-padding-32">
    <p>Dayananda Sagar College of Engineering</p>
</footer>

</body>
</html>

'''

result_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Pneumonia Detector - Result</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <style>
        body {font-family: "Times New Roman", Georgia, Serif;}
        h1, h2, h3, h4, h5, h6 {
            font-family: "Playfair Display";
            letter-spacing: 5px;
        }
        .custom-file-upload, input[type="submit"], .result-button {
            background-color: #888;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            display: inline-block;
        }
        .custom-file-upload:hover, input[type="submit"]:hover, .result-button:hover {
            background-color: #555;
        }
        .container {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center; /* Vertically center content */
        align-items: center; /* Horizontally center content */
        height: 100%; /* Ensure it takes full height */
        margin-top: 30px;
    }
        img {
            max-width: 40%;
            height: auto;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        }
        .result-button {
            text-decoration: none; /* Remove underline */
        }
    </style>
</head>
<body>

<!-- Navbar (sit on top) -->
<div class="w3-top">
    <div class="w3-bar w3-white w3-padding w3-card" style="letter-spacing:4px;">
        <a href="#" class="w3-bar-item">Pneumonia Detector</a>
    </div>
</div>

<!-- Header -->
<header class="w3-display-container w3-content w3-wide" style="max-width:1600px;min-width:500px" id="home">
    <div class="w3-display-bottomleft w3-padding-large w3-opacity">
        <h1 class="w3-xxlarge">Analysis Result</h1>
    </div>
</header>

<!-- Page content -->
<div class="w3-content" style="max-width:1100px">

    <!-- Result Section -->
    <div class="w3-container w3-padding-64" id="result">
        <div class="container">
            <h1>Prediction Result</h1>
            <p style="font-size: 1.5em;">{{ result }}</p>
            <img src="{{ image_path }}" alt="X-Ray Image">
            <p>Disclaimer: Please note that this analysis is done by an AI model and may not be 100% accurate.</p>
            <a href="/" class="result-button">Upload Another Image</a>
        </div>
    </div>

 <hr>

    <!-- Contact Section -->
    <div class="w3-container w3-padding-64" id="contact">
        <h1>Contact</h1><br>
        <p>If you have any questions or need further information, feel free to contact us.</p>
        <p class="w3-text-blue-grey w3-large"><b></b></p>
        <p>You can contact us by email at user@gmail.com, or you can send us a message here:</p>
        <form action="/action_page.php" target="_blank">
            <p><input class="w3-input w3-padding-16" type="text" placeholder="Name" required name="Name"></p>
            <p><input class="w3-input w3-padding-16" type="number" placeholder="How many people" required name="People"></p>
            <p><input class="w3-input w3-padding-16" type="datetime-local" placeholder="Date and time" required name="date" value="2020-11-16T20:00"></p>
            <p><input class="w3-input w3-padding-16" type="text" placeholder="Message / Special requirements" required name="Message"></p>
            <p><button class="w3-button w3-light-grey w3-section" type="submit">SEND MESSAGE</button></p>
        </form>
    </div>

<!-- End page content -->
</div>

<!-- Footer -->
<footer class="w3-center w3-light-grey w3-padding-32">
    <p>Dayananda Sagar College of Engineering</p>
</footer>

</body>
</html>
'''





if __name__ == '__main__':
    os.makedirs('/content/uploads', exist_ok=True)

    public_url = ngrok.connect(5000, bind_tls=True)
    print(" * ngrok URL:", public_url)

    app.run(port=5000)
    
    