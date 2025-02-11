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

from app import manager

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
        os.remove(image_path)
        return "data:image/jpeg;base64," + base64_string
    else:
        print("Error: Image or logo not loaded. Check the file paths.")

def analyse_images(question, images):
    # API headers for authentication and content type
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv('AZURE_OPENAI_API_KEY'),
    }

    all_images_analysis = []
    for image in images:
        # Construct the payload for the current image and question
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
                            "text": "\n"  # Blank line to separate sections
                        },
                        {
                            "type": "image_url",  # Sending the image URL or base64 encoded image
                            "image_url": {
                                "url": image  # The image URL or base64 string
                            }
                        },
                        {
                            "type": "text",
                            "text": question  # The evaluation question (e.g., does the image align with the prompt)
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800
        }

        # Endpoint for the image analysis API
        ENDPOINT = os.getenv('ENDPOINT')

        # Send the POST request to the image analysis API
        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error if the response is not successful
        except requests.RequestException as e:
            print(f"Error during image analysis: {e}")
            continue  # Move to the next image

        # Process the response and extract the evaluation result
        try:
            output = response.json()
            result = output['choices'][0]['message']['content']
            all_images_analysis.append(result)
        except KeyError:
            print(f"Error: Unexpected response format.")
            all_images_analysis.append('')

    return all_images_analysis
