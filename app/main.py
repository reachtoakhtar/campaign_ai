import base64
import io
import json
import os
import smtplib
import uuid
from email.message import EmailMessage
from typing import List, Optional

import cv2
import requests
import numpy as np
from PIL import Image
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from app import manager
from app.helpers import send_email, process_logo
from app.nodes import GraphState, prompt_generation_node, image_generation_node, image_grade_node, \
    image_generation_feedback_node, mail_caption_subject_generation_node, image_evaluation_node

########################
# Fast API configuration
########################
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#######################

load_dotenv()

# Directory to temporarily store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vectorstore = Chroma(
    embedding_function=AzureOpenAIEmbeddings(model=os.getenv('AZURE_OPENAI_TEXT_DEPLOYMENT_NAME'),
    api_key=os.getenv('AZURE_OPENAI_TEXT_API_KEY'),
    azure_endpoint= os.getenv('AZURE_OPENAI_TEXT_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_TEXT_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_TEXT_VERSION'),
    ),
    persist_directory="./chroma_db",
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

MAX_GENERATIONS = 3

@app.websocket("/communicate")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            images = {'accepted': [], 'rejected': []}
            input_text = ''
            target_audiences = []
            features = []
            image_resolutions = []
            try:
                data = await websocket.receive_json()
                input_text = data.get("prompt")
                target_audiences = data.get('targetAudiences')
                features = data.get('features')
                image_resolutions = data.get('imageResolutions')
            except Exception:
                print('Data Error.')
                await manager.send_response({'response': 'Data Error.'}, websocket)

            await manager.send_response({'response': 'Connected to server'}, websocket)


            pipeline = StateGraph(GraphState)

            pipeline.add_node('prompt_generation_node', prompt_generation_node)
            pipeline.add_node('image_generation_node', image_generation_node)
            pipeline.add_node('image_grade_node', image_grade_node)
            pipeline.add_node('image_generation_feedback_node', image_generation_feedback_node)
            pipeline.add_node('mail_param_generation_node', mail_caption_subject_generation_node)

            pipeline.add_edge(START, 'prompt_generation_node')
            pipeline.add_edge('prompt_generation_node', 'image_generation_node')
            pipeline.add_edge('image_generation_node', 'image_grade_node')
            pipeline.add_conditional_edges(
                'image_grade_node',
                image_evaluation_node,
                {
                    "useful": 'mail_param_generation_node',
                    "not relevant": 'image_generation_feedback_node',
                    "max_image_generation_reached": END

                }
            )
            pipeline.add_edge('image_generation_feedback_node', 'prompt_generation_node')
            pipeline.add_edge('mail_param_generation_node', END)

            rag_pipeline = pipeline.compile()
            inputs = {
                'websocket': websocket,
                'user_prompt': input_text,
                'image_resolutions': image_resolutions,
                'target_audiences': target_audiences,
                'features': features
            }

            outputs = await rag_pipeline.ainvoke(inputs)
            print('outputs')
            print(outputs)

            # await manager.send_response({"images": images, "mailSubject": mail_sub, "mailContent": mail_cont}, websocket)
            # return JSONResponse(content={"image": image_stream, "mailSubject": mail_sub, "mailContent": mail_cont}, status_code=200)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        response = {"status_code": 500, "error": f"Something went wrong."}
        await manager.send_response(response, websocket)
    except Exception as e:
        await manager.send_response({"error":str(e)}, websocket)


@app.post("/file-process/")
async def file_process(file: UploadFile = File(...)):
    llm_engine = AzureChatOpenAI(
        model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        temperature=0.7
    )

    file_ext = file.filename.split('.')[1]
    file_name = f'file_{uuid.uuid4()}.{file_ext}'
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    with open(file_path, "wb") as temp_file:
        temp_file.write(await file.read())

    file_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    document = Document(file_path)
    docx_text = "\n".join([paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()])

    # Split the text into chunks
    chunks = file_text_splitter.split_text(docx_text)
    combined_text = " ".join(chunks)

    # First, identify the product type, categories, and target audience
    category_messages = [
        {"role": "system", "content": """Analyze the text and identify:
          1. The type of product being described
          2. The most relevant 4-6 main feature categories for this type of product
          3. The most relevant 4-6  target market audience for this type of product

          Return ONLY a JSON object in this exact format:
          {
              "product_type": "type",
              "categories": ["category1", "category2", "category3"],
              "target_audience": ["audience1", "audience2"]
          }"""},
        {"role": "user", "content": combined_text}
    ]

    category_response = llm_engine.generate([category_messages])

    # Clean and parse JSON response
    try:
        response_text = category_response.generations[0][0].text.strip()
        # Remove any markdown code block indicators if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        product_info = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {response_text}")
        raise e

    # Now get summaries based on identified categories and audience
    summary_messages = [
        {"role": "system", "content": f"""For this {product_info['product_type']}, provide:
           1. A brief highlight for each of these categories: {', '.join(product_info['categories'])}
           2. Key characteristics of the target audience: {', '.join(product_info['target_audience'])}

           Return ONLY a JSON object in this format:
           {{
               "features": {{"category1": "highlight1", "category2": "highlight2"}},
               "audience_characteristics": {{"audience1": "characteristic1", "audience2": "characteristic2"}}
           }}"""},
        {"role": "user", "content": combined_text}
    ]

    summary_response = llm_engine.generate([summary_messages])

    # Clean and parse JSON response
    try:
        response_text = summary_response.generations[0][0].text.strip()
        # Remove any markdown code block indicators if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        analysis_results = json.loads(response_text)
        os.remove(file_path)

        # Combine all results into a final output
        final_analysis = {
            "product_type": product_info["product_type"],
            "feature_summary": analysis_results["features"],
            "target_audience": {
                "segments": product_info["target_audience"],
                "characteristics": analysis_results["audience_characteristics"]
            }
        }
        return final_analysis
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {response_text}")
        raise e


@app.post("/logo-process/")
async def logo_process(logo: UploadFile = File(...), image: str = Form(...)):
    try:
        logo_ext = logo.filename.split('.')[1]
        logo_name = f'logo_{uuid.uuid4()}.{logo_ext}'
        logo_path = os.path.join(UPLOAD_FOLDER, logo_name)
        with open(logo_path, "wb") as temp_file:
            temp_file.write(await logo.read())

        base64_code = image.split(',')[1]
        img_data = base64_code.encode()
        content = base64.b64decode(img_data)
        image_name = f'image_{uuid.uuid4()}.png'
        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        with open(image_path, 'wb') as fw:
            fw.write(content)

        superimposed_image = process_logo(logo_path, image_path)
        return JSONResponse(content={"image": superimposed_image}, status_code=200)
    except Exception as e:
        print(repr(e))
        raise HTTPException(status_code=500, detail=f"Failed to process logo: {str(e)}")


@app.post("/send-mail/")
async def send_mail(subject: str = Form(...), body: str = Form(...), image: str = Form(...)):
    try:
        send_email(subject, body, image)
        return JSONResponse(content={"message": "Mail sent successfully"}, status_code=200)
    except Exception as e:
        # Clean up and handle errors
        print(repr(e))
        raise HTTPException(status_code=500, detail=f"Failed to send mail: {str(e)}")
