__author__ = "akhtar"

import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

load_dotenv()

llm_engine = AzureChatOpenAI(
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    temperature=0.7
)

dalle_client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_DALLE_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_DALLE_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_DALLE_ENDPOINT')
)
