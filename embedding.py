"""
Unified Embedding Client
"""

import os
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

endpoint = "https://models.github.ai/inference"
model_name = "openai/text-embedding-3-large"

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

class Embedding:
    def __init__(self):
        self.client = EmbeddingsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token)
        )

    def embed(self, inputs):
        """
        inputs: str or list[str]
        returns the embeddings data from the API response
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        response = self.client.embed(
            input=inputs,
            model=model_name
        )
        return response.data
