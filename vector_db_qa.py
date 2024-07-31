# {{{ imports 

import pandas as pd
import numpy as np

from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.utilities import ApifyWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import AzureSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import matplotlib.pyplot as plt

import os
import json
from typing import Iterable
from dotenv import load_dotenv
load_dotenv()
# }}} 
# {{{ env variables 

APIFY_API_TOKEN = os.getenv('APIFY_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = '2023-05-15'
AZURE_OPENAI_API_VERSION = '2023-05-15'
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_COGNITIVE_SEARCH_ENDPOINT = os.getenv('AZURE_COGNITIVE_SEARCH_ENDPOINT')
AZURE_COGNITIVE_SEARCH_API_KEY = os.getenv('AZURE_COGNITIVE_SEARCH_API_KEY')
AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv('AZURE_COGNITIVE_SEARCH_INDEX_NAME')
model: str = "text-embedding-ada-002"
# }}} 
# {{{ bob website data 

# apify = ApifyWrapper()
# loader = apify.call_actor(
#     actor_id='apify/website-content-crawler',
#     run_input={'startUrls':[{'url': 'https://www.bankofbaroda.in/'}]},
#     dataset_mapping_function=lambda item: Document(page_content=item["text"] or "", metadata={"source": item["url"]}),
# )
# }}}
# {{{ load data
def load_docs(file_path):
    docs = []
    with open(file_path, 'r') as jsonl_data:
        for line in jsonl_data:
            data = json.loads(line)
            obj = Document(**data)
            docs.append(obj)

    return docs

# }}}
docs = load_docs('./bob_website_data.jsonl')

# {{{ plot chunks 

docs_length = []
for i in range(len(docs)):
    docs_length.append(len(docs[i].page_content))

print(f'doc lengths\nmin: {min(docs_length)} \navg.: {round(np.average(docs_length), 1)} \nmax: {max(docs_length)}')

# plt.figure(figsize=(15, 3))
# plt.plot(docs_length, marker='o')
# plt.title("doc length")
# plt.ylabel("# of characters")
# plt.show()
# }}} 
# {{{ chunk docs 

chunk_size = 700
chunk_overlap = 150

def chunk_docs(doc, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len)
    chunks = text_splitter.create_documents(texts=[doc.page_content], metadatas=[{'source': doc.metadata['source']}])
    return chunks

chunked_docs = []

for i in docs:
    chunked_docs.append(chunk_docs(i, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
flattened_chunked_docs = [doc for docs in chunked_docs for doc in docs]
# }}} 
# {{{ initialize Azure embeddings, vector DB and add data 

embeddings = AzureOpenAIEmbeddings(
    azure_deployment='RG210-openai-ada',
    openai_api_version=OPENAI_API_VERSION,
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)

index_name = 'bob-data-langchain-index'
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_COGNITIVE_SEARCH_API_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

# vector_store.add_documents(documents=flattened_chunked_docs)
# query_docs = vector_store.similarity_search(
#     query = 'what is the data about',
#     k=5,
#     search_type='similarity'
# )
# }}} 
llm = AzureChatOpenAI(deployment_name='RG210-openai-35turbo', openai_api_key=AZURE_OPENAI_API_KEY, temperature=0.1, api_version = OPENAI_API_VERSION)
# {{{ initialize prompt and qa chain 

prompt_template = """You are an expert financial advisor from Bank of Baroda, assisting users with specific Bank of Baroda products based on their financial data and user profile.
Context:
{context}
Chat history:
{chat_history}
Query: {question}
"""
prompt = PromptTemplate(template = prompt_template, inputVariables = ["chat_history", "question"])

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    # condense_question_prompt = PromptTemplate.from_template(prompt_template),
    combine_docs_chain_kwargs = {"prompt": prompt},
    return_source_documents = True,
    verbose=True
)

chat_history = []
# }}} 

def query_vector_db(query):
    qa_result = qa({
        "question": query,
        "chat_history": chat_history,
    })

    sources = []
    for i in qa_result['source_documents']:
        sources.append(i.metadata['source'])
    
    qa_result = {'answer': qa_result['answer'], "sources": sources}
    
    return qa_result


#TODO: increase azure ai search quota to index entire data
