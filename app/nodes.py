__author__ = "akhtar"

import os
from typing import List, Optional, ClassVar

from dotenv import load_dotenv
from fastapi import WebSocket
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app import manager
from app.helpers import process_image_to_base64, image_analysis
from app.llm import llm_engine, dalle_client

load_dotenv()

MAX_GENERATIONS = 3

class GraphState(BaseModel):
    websocket: 'WebSocket'
    user_prompt: Optional[str] = None
    image_resolutions: List[dict[str, int]] = []
    target_audiences: List[str] = []
    features: List[dict[str, str]] = []

    generation: Optional[str] = None
    documents: List[str] = []
    image: Optional[str] = None
    imageAnalysis: Optional[str] = None
    imageGeneratefeedbacks: Optional[str] = None
    imageGenerationNum: int = 0
    mail_subject: Optional[str] = None
    mail_content: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class GradeAnalysis(BaseModel):
    grade: str = Field(
        description="It gives yes / no if the image aligns with prompt."
    )
    reason: str = Field(
        description="Reason on why the image is not up to the mark."
    )


class MailParams(BaseModel):
    subject: str = Field(
        description="Subject for mail."
    )
    content: str = Field(
        description="Content body for mail"
    )


def prompt_generation_node(state: GraphState):
    system_prompt = """
                  You are an assistant for creating prompt for dall-e-3 model to create car image for sale based on the user prompt and retrieved context.
                  The retrieved context can be blank and hence in that case remember to use only userPrompt for creating prompt for dall-e-3 model to create car image.
                  Additional feedback may be provided for why a previous version of the prompts didn't lead to a valid response. Make sure to utilize that feedback to generate a better prompt for dall-e-3 model.
                  Only provide the prompt for attractive image that does not include any text in the image and nothing else!
                  """

    human_prompt = """
              userPrompt: {userPrompt}

              Here is the additional feedback about previous versions of the prompts:
              {feedback}

              Context:
              {context}

              Answer:
              """

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    rag_chain = rag_prompt | llm_engine | StrOutputParser()
    target_aud = []
    try:
        target_aud = state.target_audiences[0]
    except IndexError:
        pass

    generation = rag_chain.invoke({
        "context": "\n\n".join(state.documents).join(target_aud),
        "userPrompt": state.user_prompt,
        "feedback": state.imageGeneratefeedbacks
    })
    return {"generation": generation}


async def image_generation_node(state: GraphState):
    welcome = 'Generating the image. Please wait.'
    print(welcome)
    await manager.send_response({'response': welcome}, state.websocket)

    response = dalle_client.images.generate(
        model=os.getenv('AZURE_OPENAI_DALLE_DEPLOYMENT_NAME'),
        prompt=state.generation,
        size="1024x1024",
    )
    base64_image = process_image_to_base64(response.data[0].url, state.image_resolutions[0])
    await manager.send_response({'response': 'Image successfully generated.'}, state.websocket)
    return {"image": base64_image, "imageGenerationNum": state.imageGenerationNum + 1}


async def image_grade_node(state: GraphState):
    question = f"""
    The LLM-generated image is produced based on the image prompt, which is derived from the userPrompt and a set of retrieved facts. There is no human in the loop to access or modify the LLM-generated image.
    Your role is to act as an evaluator, evaluate if the LLM-generated image aligns with the userPrompt. Please provide with 'yes' or 'no' based on the following criteria:
    1. Alignment with the userPrompt:
        Assess how closely the image matches the userPrompt given to the LLM.
    ***Evaluation guide:
    -> 'yes' means the image perfectly aligns with the userPrompt.
    -> 'no' means the image shows minimal or no alignment with the userPrompt.
    ***
    Be honest and objective in your evaluation. Your evaluation should reflect how well the image meets the criteria overall.
    ***REMEBER i only want 'yes' or 'no' and a reason why...if the image is not at all matching then return 'no' else 'yes' ***
    This is the userPrompt:
    {state.user_prompt}
    """
    print('inside image anaysis')
    output = image_analysis(question, state.image)
    return {"imageAnalysis": output}


async def image_evaluation_node(state: GraphState):
    welcome = "Evaluating the generated image."
    print(welcome)
    await manager.send_response({'response': welcome}, state.websocket)

    system_prompt = """
       You are an AI assistant that gives 'yes' or 'no' along with the reason from the image analysis provided. Don't formulate the reason,
       just give the reason provided in the image analysis input.
       """
    human_prompt = """
       This is the image analysis:
       {imageAnalysis}
       """
    image_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    image_analysis_grader = (
        image_analysis_prompt
        | llm_engine.with_structured_output(GradeAnalysis)
    )

    # Invoke the image analysis grader
    image_analysis_grade = image_analysis_grader.invoke(
        {"imageAnalysis": state.imageAnalysis}
    )
    if image_analysis_grade.grade.lower() == 'yes':
        # images['accepted'].append(state.image)
        return "useful"
    elif state.imageGenerationNum > MAX_GENERATIONS:
        # images['accepted'].append(state.image)
        return "max_image_generation_reached"
    else:
        # images['rejected'].append(state.image)
        # await manager.send_response({"rejected_images": images['rejected']}, state.websocket)
        return "not relevant"


async def image_generation_feedback_node(state: GraphState):
    welcome = "Image generation feedback node."
    print(welcome)
    await manager.send_response({'response': welcome}, state.websocket)

    question = f"""
    Your role is to give feedback on the LLM generated image that is in base64 encoded. The LLM generated image(base64 encoded) is aligned with the userPrompt.
    Explain how the LLM generated image could be improved so that it aligns with the userPrompt.
    Only provide your feedback and nothing else!
    This is the userPrompt:
    {state.user_prompt}
    """

    output = image_analysis(question, state.image)

    feedback = 'Feedback about the image : {}'.format(
        output
    )
    print('feedback-----', feedback)
    return {"imageGeneratefeedbacks": feedback}


def mail_caption_subject_generation_node(state: GraphState):
    print("Generating email caption.")

    system_prompt = """
    Your role is to general subject and content body for mail based on the LLM generated image(base64 encoded) and userPrompt.
    In the content body dont include "please reply to this email or contact us" as it will be a system generated mail.
    Include the context and the user prompt to generate the content body for the mail.
    Only provide subject and content body for mail and nothing else!
    """

    human_prompt = """
    This is the userPrompt:
    {userPrompt}
    LLM generated image: {image}
    context: {context}
    """

    mail_caption_subject_generation_feedback_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    mail_caption_subject_generation_feedback_chain = (
        mail_caption_subject_generation_feedback_prompt
        | llm_engine.with_structured_output(MailParams)
    )

    mail_params = mail_caption_subject_generation_feedback_chain.invoke({
        "userPrompt": state.user_prompt,
        "image": state.image,
        "context": state.features.extend(state.target_audiences)
    })

    return {"mail_subject": mail_params.subject, "mail_content": mail_params.content}
