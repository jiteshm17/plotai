import os
import openai

from dotenv import load_dotenv
load_dotenv()


class ChatGPT():

    temperature = 0
    max_tokens = 1000
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0.6
    model = "gpt-3.5-turbo"
    engine = "gpt35turbo16k"


    def __init__(self):
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    @property
    def _default_params(self):
        return {
            "engine":self.engine,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            # "model": self.model,
        }

    def chat(self, prompt):
        params = {
            **self._default_params,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                }
            ],
        }
        response = openai.ChatCompletion.create(**params)
        return response["choices"][0]["message"]["content"]
