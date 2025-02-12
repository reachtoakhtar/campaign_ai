__author__ = "akhtar"

import base64
import io
import os
import smtplib
from email.message import EmailMessage

import cv2
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def process_image_to_base64(url, image_resolution):
    response = requests.get(url)
    image_bytes = io.BytesIO(response.content)
    img = Image.open(image_bytes)
    width = image_resolution['width']
    height = image_resolution['height']
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    jpeg_image = io.BytesIO()
    img.save(jpeg_image, format='JPEG')
    jpeg_image.seek(0)
    base64_string = base64.b64encode(jpeg_image.read()).decode('utf-8')
    return "data:image/jpeg;base64," + base64_string

def send_email(subject, body, image,):
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port= os.getenv('SMTP_PORT')
    email_from = os.getenv("EMAIL_FROM")
    email_to = os.getenv("EMAIL_TO")
    email_password = os.getenv('EMAIL_PASSWORD')

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = email_from
    msg['To'] = email_to
    msg.set_content(body)

    try:
        base64_code = image.split(',')[1]
        img_data = base64_code.encode()
        content = base64.b64decode(img_data)

        with open('../image.png', 'wb') as fw:
            fw.write(content)

        with open('../image.png', 'rb')  as img_file:
            img_data = img_file.read()
            img_type = 'png'
            img_name = 'image.png'
            msg.add_attachment(img_data, maintype='image', subtype=img_type, filename=img_name)

        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(email_from, email_password)
            server.send_message(msg)
        return "Success"
    except Exception as e:
        print(f'Error: {e}')
        return "Error"

def process_logo(logo_path, image_path):
    image = cv2.imread(image_path)
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)  # Ensure transparency if the logo has alpha channel

    # Check if images are loaded
    if image is not None and logo is not None:
        # Resize the logo
        desired_logo_width = 75  # Adjust this value (e.g., 50 to 100)
        scale_ratio = desired_logo_width / logo.shape[1]
        new_logo_size = (desired_logo_width, int(logo.shape[0] * scale_ratio))
        resized_logo = cv2.resize(logo, new_logo_size, interpolation=cv2.INTER_AREA)

        # Get the top-right corner position
        padding = 10  # Space from the edge
        x_offset = image.shape[1] - resized_logo.shape[1] - padding
        y_offset = padding

        # Overlay the logo on the main image
        # Check if the logo has an alpha channel (transparency)
        if resized_logo.shape[2] == 4:  # RGBA
            for c in range(0, 3):  # Iterate over RGB channels
                image[y_offset:y_offset + resized_logo.shape[0], x_offset:x_offset + resized_logo.shape[1], c] = \
                    image[y_offset:y_offset + resized_logo.shape[0], x_offset:x_offset + resized_logo.shape[1], c] * \
                    (1 - resized_logo[:, :, 3] / 255.0) + resized_logo[:, :, c] * (resized_logo[:, :, 3] / 255.0)
        else:
            # If no alpha channel, simply overlay
            image[y_offset:y_offset + resized_logo.shape[0], x_offset:x_offset + resized_logo.shape[1]] = resized_logo

        base64_string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        os.remove(logo_path)
        os.remove(image_path)
        return "data:image/jpeg;base64," + base64_string
    else:
        print("Error: Image or logo not loaded. Check the file paths.")

def image_analysis(question, encoded_image):
    # Configuration
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv('AZURE_OPENAI_API_KEY'),
    }
    # Payload for the request
    payload = {
      "messages": [
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are an AI assistant that helps people find information."
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "\n"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": encoded_image
              }
            },
            {
              "type": "text",
              "text": question
            }
          ]
        }
      ],
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 800
    }
    ENDPOINT = os.getenv('ENDPOINT')
    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    output = response.json()
    return output['choices'][0]['message']['content']
