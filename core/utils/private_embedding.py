from langchain_community.embeddings import OllamaEmbeddings
import os

# os.environ["OPENAI_API_KEY"]=None #your openai api key

# this is a default model for debugging purposes
embeddings = OllamaEmbeddings() 

# from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large") #refer to https://beta.openai.com/docs/guides/engines/text-embedding-3-large for more information


