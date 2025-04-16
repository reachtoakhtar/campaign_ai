import ast
import base64
import json
import os
import uuid
from typing import Optional

from docx import Document
from dotenv import load_dotenv
from fastapi import (
    FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START

from app import manager
from app.helpers import send_email, process_logo
from app.llm import llm_engine
from app.nodes import (
    GraphState, prompt_generation_node, image_generation_node, image_grade_node, image_generation_feedback_node,
    image_evaluation_node, MailParams, IMAGES,
)

########################
# Fast API configuration
########################
app = FastAPI ()

app.add_middleware (
    CORSMiddleware, allow_origins=["*"],  # Allow all origins or specify your React app's URL
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )
#######################

load_dotenv ()

# Directory to temporarily store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs (UPLOAD_FOLDER, exist_ok=True)

vectorstore = Chroma (
    embedding_function=AzureOpenAIEmbeddings (
        model=os.getenv ('AZURE_OPENAI_TEXT_DEPLOYMENT_NAME'), api_key=os.getenv ('AZURE_OPENAI_TEXT_API_KEY'),
        azure_endpoint=os.getenv ('AZURE_OPENAI_TEXT_ENDPOINT'), azure_deployment=os.getenv (
            'AZURE_OPENAI_TEXT_DEPLOYMENT_NAME'
        ), openai_api_version=os.getenv ('AZURE_OPENAI_API_TEXT_VERSION'), ), persist_directory="./chroma_db", )
text_splitter = RecursiveCharacterTextSplitter (
    chunk_size=500, chunk_overlap=50, )


@app.websocket ("/communicate")
async def websocket_endpoint (websocket: WebSocket):
    await manager.connect (websocket)
    try:
        while True:
            input_text = ''
            target_audiences = []
            features = []
            image_resolutions = []
            try:
                data = await websocket.receive_json ()
                input_text = data.get ("prompt")
                target_audiences = data.get ('targetAudiences')
                features = data.get ('features')
                image_resolutions = data.get ('imageResolutions')
            except Exception:
                print ('Data Error.')
                await manager.send_response (
                    {'response': 'Data Error.'}, websocket
                )

            await manager.send_response (
                {'response': 'Connected to server'}, websocket
            )

            pipeline = StateGraph (GraphState)

            pipeline.add_node ('prompt_generation_node', prompt_generation_node)
            pipeline.add_node ('image_generation_node', image_generation_node)
            pipeline.add_node ('image_grade_node', image_grade_node)
            pipeline.add_node (
                'image_generation_feedback_node', image_generation_feedback_node
            )

            pipeline.add_edge (START, 'prompt_generation_node')
            pipeline.add_edge (
                'prompt_generation_node', 'image_generation_node'
            )
            pipeline.add_edge ('image_generation_node', 'image_grade_node')
            pipeline.add_conditional_edges (
                'image_grade_node', image_evaluation_node, {
                    "useful": END, "max_image_generation_reached": END, "irrelevant": 'image_generation_feedback_node',
                }
            )
            pipeline.add_edge (
                'image_generation_feedback_node', 'prompt_generation_node'
            )

            rag_pipeline = pipeline.compile ()

            images_per_target = {}
            for target_audience in target_audiences:
                inputs = {
                    'websocket': websocket, 'user_prompt': input_text, 'image_resolutions': image_resolutions,
                    'target_audience': target_audience, 'features': features
                }

                output = await rag_pipeline.ainvoke (inputs)
                images_per_target[target_audience] = IMAGES[target_audience]

            await manager.send_response ({"images_per_target": images_per_target}, websocket)

    except WebSocketDisconnect:
        manager.disconnect (websocket)
        response = {"status_code": 500, "error": f"Something went wrong."}
        await manager.send_response (response, websocket)
    except Exception as e:
        await manager.send_response ({"error": str (e)}, websocket)


@app.post ("/file-process/")
async def file_process (prompt: Optional[str] = Form (''), file: Optional[UploadFile] = File (None)):
    llm_engine = AzureChatOpenAI (
        model=os.getenv ('AZURE_OPENAI_DEPLOYMENT_NAME'), api_key=os.getenv ('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv ('AZURE_OPENAI_ENDPOINT'), azure_deployment=os.getenv ('AZURE_OPENAI_DEPLOYMENT_NAME'),
        openai_api_version=os.getenv ('AZURE_OPENAI_API_VERSION'), temperature=0.7
    )

    if len(prompt):
        docx_text = prompt
    else:
        file_ext = file.filename.split ('.')[1]
        file_name = f'file_{uuid.uuid4 ()}.{file_ext}'
        global file_path
        file_path = os.path.join (UPLOAD_FOLDER, file_name)
        with open (file_path, "wb") as temp_file:
            temp_file.write (await file.read ())

        document = Document (file_path)
        docx_text = "\n".join (
            [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip ()]
        )

    # Split the text into chunks
    file_text_splitter = RecursiveCharacterTextSplitter (
        chunk_size=500, chunk_overlap=50, )
    chunks = file_text_splitter.split_text (docx_text)
    combined_text = " ".join (chunks)

    # First, identify the product type, categories, and target audience
    category_messages = [{
        "role": "system", "content": """Analyze the text and identify:
          1. The type of product being described
          2. The most relevant 4-6 main feature categories for this type of product
          3. The most relevant 4-6  target market audience for this type of product

          Return ONLY a JSON object in this exact format:
          {
              "product_type": "type",
              "categories": ["category1", "category2", "category3"],
              "target_audience": ["audience1", "audience2"]
          }"""
    }, {"role": "user", "content": combined_text}]

    category_response = llm_engine.generate ([category_messages])

    # Clean and parse JSON response
    try:
        response_text = category_response.generations[0][0].text.strip ()
        # Remove any markdown code block indicators if present
        response_text = response_text.replace ('```json', '').replace (
            '```', ''
        ).strip ()
        product_info = json.loads (response_text)
    except json.JSONDecodeError as e:
        print (f"Error parsing JSON: {response_text}")
        raise e

    # Now get summaries based on identified categories and audience
    summary_messages = [{
        "role": "system", "content": f"""For this {product_info['product_type']}, provide:
           1. A brief highlight for each of these categories: {', '.join (product_info['categories'])}
           2. Key characteristics of the target audience: {', '.join (product_info['target_audience'])}

           Return ONLY a JSON object in this format:
           {{
               "features": {{"category1": "highlight1", "category2": "highlight2"}},
               "audience_characteristics": {{"audience1": "characteristic1", "audience2": "characteristic2"}}
           }}"""
    }, {"role": "user", "content": combined_text}]

    summary_response = llm_engine.generate ([summary_messages])

    # Clean and parse JSON response
    try:
        response_text = summary_response.generations[0][0].text.strip ()
        # Remove any markdown code block indicators if present
        response_text = response_text.replace ('```json', '').replace (
            '```', ''
        ).strip ()
        analysis_results = json.loads (response_text)
        if file is not None:
            os.remove (file_path)

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
        print (f"Error parsing JSON: {response_text}")
        raise e


@app.post ("/logo-process/")
async def logo_process (images = Form(...), logo: UploadFile = File (...)):
    try:
        logo_ext = logo.filename.split ('.')[1]
        logo_name = f'logo_{uuid.uuid4 ()}.{logo_ext}'
        logo_path = os.path.join (UPLOAD_FOLDER, logo_name)

        with open (logo_path, "wb") as f:
            f.write(await logo.read ())

        target_images = ast.literal_eval (images)
        accepted_images = target_images.get('accepted')
        rejected_images = target_images.get('rejected')
        response = {'accepted': [], 'rejected': []}

        for image in accepted_images:
            base64_code = image.split (',')[1]
            img_data = base64_code.encode ()
            content = base64.b64decode (img_data)
            image_name = f'image_{uuid.uuid4 ()}.png'
            image_path = os.path.join (UPLOAD_FOLDER, image_name)
            with open (image_path, 'wb') as fw:
                fw.write (content)

            superimposed_image = process_logo (logo_path, image_path)
            response['accepted'].append(superimposed_image)

        for image in rejected_images:
            base64_code = image.split (',')[1]
            img_data = base64_code.encode ()
            content = base64.b64decode (img_data)
            image_name = f'image_{uuid.uuid4 ()}.png'
            image_path = os.path.join (UPLOAD_FOLDER, image_name)
            with open (image_path, 'wb') as fw:
                fw.write (content)

            superimposed_image = process_logo (logo_path, image_path)
            response['rejected'].append (superimposed_image)

        os.remove(logo_path)
        return JSONResponse (
            content=response, status_code=200
        )
    except Exception as e:
        print (repr (e))
        raise HTTPException (
            status_code=500, detail=f"Failed to process logo: {str (e)}"
        )


@app.post ("/generate-email/")
async def generate_mail (
    user_prompt: str = Form (...), features = Form (...), target_audience: str = Form (...)
):
    try:
        system_prompt = """
        Your role is to general subject and content body for mail based on the user_prompt and given context.
        In the content body dont include "please reply to this email or contact us" as it will be a system generated 
        mail.
        Include the context and the user prompt to generate the content body for the mail.
        Only provide subject and content body for mail and nothing else!
        """

        human_prompt = """
        This is the user_prompt:
        {user_prompt}
        context: {context}
        """
        mail_caption_subject_generation_feedback_prompt = ChatPromptTemplate.from_messages (
            [("system", system_prompt), ("human", human_prompt), ]
        )

        mail_caption_subject_generation_feedback_chain = (
                mail_caption_subject_generation_feedback_prompt | llm_engine.with_structured_output (MailParams))

        feature_list = ast.literal_eval (features)
        mail_params = mail_caption_subject_generation_feedback_chain.invoke (
            {
                "user_prompt": user_prompt, "context": feature_list.extend (
                [{'target_audience': target_audience}]
            )
            }
        )

        response = {
            "mail_subject": mail_params.subject, "mail_content": mail_params.content
        }

        return JSONResponse (content=response, status_code=200)
    except Exception as e:
        # Clean up and handle errors
        print (repr (e))
        raise HTTPException (
            status_code=500, detail=f"Failed to generate mail: {str (e)}"
        )


@app.post ("/send-mail/")
async def send_mail (
    subject: str = Form (...), body: str = Form (...), image: str = Form (...)
):
    try:
        send_email (subject, body, image)
        return JSONResponse (
            content={"message": "Mail sent successfully"}, status_code=200
        )
    except Exception as e:
        # Clean up and handle errors
        print (repr (e))
        raise HTTPException (
            status_code=500, detail=f"Failed to send mail: {str (e)}"
        )
