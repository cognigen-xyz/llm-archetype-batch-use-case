import logging
import os
from typing import Any

import dotenv
import openai
from openai import AzureOpenAI, OpenAI

#client = AzureOpenAI(api_key=os.environ["OPENAI_API_KEY"],
#api_version=os.environ["OPENAI_API_VERSION"])
import vertexai  # type: ignore
from langchain.llms import AzureOpenAI
from vertexai.language_models import TextGenerationModel  # type: ignore

dotenv.load_dotenv()

GCP_PROJECT = os.environ["GCP_PROJECT"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base=os.environ["OPENAI_API_BASE"])'
# openai.api_base = os.environ["OPENAI_API_BASE"]




logger = logging.getLogger(__name__)


def get_llm(model_name: str):
    if model_name == "gpt-4":
        llm = get_gpt_4()
    elif model_name == "gpt-3":
        llm = get_gpt_35_turbo()
    elif model_name == "palm":
        llm = get_palm()
    else:
        raise ValueError(f"Invalid model: {model_name}")

    return llm


def get_gpt_4():
    logger.info("Loading LLM 'gpt-4'")
    return AzureOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=256,
        top_p=1.0,
        verbose=False,
        engine="gpt-4-us",
    )

class OpenAIGPT35:
    def __init__(self):
        self.model = OpenAI(base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"])
    
    def __call__(self, prompt) -> Any:
        #list of models https://platform.openai.com/docs/models/gpt-3-5
        stream = self.model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            stream=True)
        
        
        #print(f"debug {response} {self.model.base_url}")
        response = ""

        for part in stream:
            add = part.choices[0].delta.content or ""
            response += add

        return response


def get_gpt_35_turbo():
    # Different approach than gpt-4, as for gpt-3 we get an error:
    # "The completion operation does not work with the specified model, gpt-35-turbo."
    logger.info("Loading LLM 'gpt-35-turbo'")

    # this allows running the returned llm with llm("your prompt")
    #return (
    #    lambda prompt: client.chat.completions.create(model="gpt-35-turbo",
    #    messages=[{"role": "user", "content": prompt}],
    #    temperature=0.7,
    #    max_tokens=256,
    #    top_p=1)
    #    .choices[0]
    #    .message.content
    #)
    return OpenAIGPT35()

def get_palm():
    logger.info("Loading LLM 'text-bison@001'")
    vertexai.init(project=GCP_PROJECT, location="us-central1")
    model = TextGenerationModel.from_pretrained("text-bison@001")
    return lambda prompt: model.predict(prompt).text
