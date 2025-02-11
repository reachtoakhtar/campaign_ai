__author__ = "akhtar"

import os
from typing import List, Optional, Dict

from dotenv import load_dotenv
from fastapi import WebSocket
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app import manager
from app.helpers import process_image_to_base64, analyse_images
from app.llm import llm_engine, dalle_client

load_dotenv ()

MAX_GENERATIONS = 1
NO_OF_IMAGES = 2

IMAGES = {}


class GraphState (BaseModel):
    websocket: 'WebSocket'
    user_prompt: Optional[str] = None
    image_resolutions: List[dict[str, int]] = []
    target_audience: str = ''
    features: List[dict[str, str]] = []

    generated_prompt: Optional[str] = ''
    generated_images: List[str] = []
    image_analysis: List[str] = []
    images_feedback: List[str] = []
    image_generation_num: int = 0

    class Config:
        arbitrary_types_allowed = True


class GradeAnalysis (BaseModel):
    grade: str = Field (
        description="It gives yes / no if the image aligns with prompt."
    )
    reason: str = Field (
        description="Reason on why the image is not up to the mark."
    )


class MailParams (BaseModel):
    subject: str = Field (
        description="Subject for mail."
    )
    content: str = Field (
        description="Content body for mail"
    )


def prompt_generation_node (state: GraphState):
    system_prompt = """
                  You are an assistant for creating prompt for dall-e-3 model to create car image for sale based on 
                  the user prompt and retrieved context.
                  The retrieved context can be blank and hence in that case remember to use only user_prompt for 
                  creating prompt for dall-e-3 model to create car image.
                  Additional feedback may be provided for why a previous version of the prompts didn't lead to a 
                  valid response. Make sure to utilize that feedback to generate a better prompt for dall-e-3 model.
                  Only provide the prompt for attractive image that does not include any text in the image and 
                  nothing else!
                  """

    human_prompt = """
              user_prompt: {user_prompt}

              Here is the additional feedback about previous versions of the prompts:
              {feedback}

              Context:
              {context}

              Answer:
              """

    rag_prompt = ChatPromptTemplate.from_messages (
        [("system", system_prompt), ("human", human_prompt), ]
    )

    rag_chain = rag_prompt | llm_engine | StrOutputParser ()

    generated_prompt = rag_chain.invoke (
        {
            "context": [{'target_audience': state.target_audience}], "user_prompt": state.user_prompt,
            "feedback": state.images_feedback
        }
    )
    return {"generated_prompt": generated_prompt}


async def image_generation_node (state: GraphState):
    welcome = 'Generating the images. Please wait.'
    print (welcome)
    await manager.send_response ({'response': welcome}, state.websocket)

    generated_images_for_audience = []
    for i in range (NO_OF_IMAGES):
        response = dalle_client.images.generate (
            model=os.getenv ('AZURE_OPENAI_DALLE_DEPLOYMENT_NAME'), prompt=state.generated_prompt, size="1024x1024", )
        base64_image = process_image_to_base64 (response.data[0].url, state.image_resolutions[0])
        generated_images_for_audience.append (base64_image)

    message = f'Images generated for "{state.target_audience}".'
    print (message)
    await manager.send_response ({'response': message}, state.websocket)

    return {"generated_images": generated_images_for_audience, "image_generation_num": state.image_generation_num + 1}


async def image_grade_node (state: GraphState):
    question = f"""
    The LLM-generated image is produced based on the image prompt, which is derived from the user_prompt and a set of 
    retrieved facts. There is no human in the loop to access or modify the LLM-generated image.
    Your role is to act as an evaluator, evaluate if the LLM-generated image aligns with the user_prompt. Please 
    provide with 'yes' or 'no' based on the following criteria:
    1. Alignment with the user_prompt:
        Assess how closely the image matches the user_prompt given to the LLM.
    ***Evaluation guide:
    -> 'yes' means the image perfectly aligns with the user_prompt.
    -> 'no' means the image shows minimal or no alignment with the user_prompt.
    ***
    Be honest and objective in your evaluation. Your evaluation should reflect how well the image meets the criteria 
    overall.
    ***REMEBER i only want 'yes' or 'no' and a reason why...if the image is not at all matching then return 'no' else 
    'yes' ***
    This is the user_prompt:
    {state.user_prompt}
    """
    output = analyse_images (question, state.generated_images)
    return {"image_analysis": output}


async def image_evaluation_node (state: GraphState):
    welcome = "Evaluating the generated images."
    print (welcome)
    await manager.send_response ({'response': welcome}, state.websocket)

    images_accepted = []
    images_rejected = []
    for idx, analysis in enumerate (state.image_analysis):
        IMAGES[state.target_audience] = {}

        system_prompt = """
           You are an AI assistant that gives 'yes' or 'no' along with the reason from the image analysis provided. 
           Don't formulate the reason,
           just give the reason provided in the image analysis input.
           """
        human_prompt = """
           This is the image analysis:
           {image_analysis}
           """
        image_analysis_prompt = ChatPromptTemplate.from_messages (
            [("system", system_prompt), ("human", human_prompt), ]
        )
        image_analysis_grader = (image_analysis_prompt | llm_engine.with_structured_output (GradeAnalysis))

        # Invoke the image analysis grader
        image_analysis_grade = image_analysis_grader.invoke (
            {"image_analysis": analysis}
        )

        if image_analysis_grade.grade.lower () == 'yes':
            images_accepted.append (state.generated_images[idx])
        else:
            images_rejected.append (state.generated_images[idx])

    try:
        IMAGES[state.target_audience]['accepted'].extend(images_accepted)
    except KeyError:
        IMAGES[state.target_audience]['accepted'] = images_accepted

    try:
        IMAGES[state.target_audience]['rejected'].extend(images_rejected)
    except KeyError:
        IMAGES[state.target_audience]['rejected'] = images_rejected

    if state.image_generation_num >= MAX_GENERATIONS:
        result = 'max_image_generation_reached'
    elif len (images_accepted) >= NO_OF_IMAGES:
        result = 'useful'
    else:
        result = 'irrelevant'

    print ("IMAGE EVALUATION RESULT: ", result)
    return result


async def image_generation_feedback_node (state: GraphState):
    question = f"""
        Your role is to give feedback on the LLM generated images that is in base64 encoded. The LLM generated images (
        base64 encoded) is aligned with the user_prompt and given target audience.
        Explain how the LLM generated image could be improved so that it aligns with the user_prompt and given target 
        audience.
        Only provide your feedback and nothing else!
        This is the user_prompt:
        {state.user_prompt}
        This is the target audience:
        {state.target_audience}
        """
    feedback_for_images = analyse_images (question, state.generated_images)
    return {"images_feedback": feedback_for_images}
