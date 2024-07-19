from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
# from azure.ai.openai import OpenAIClient
import pandas as pd
import io
import time
import json
import requests
from dotenv import load_dotenv
load_dotenv()

container_name = "bankstatements"
connection_string = os.getenv('connection_string')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
# openai_client = OpenAIClient(endpoint=AZURE_OPENAI_ENDPOINT, credential=DefaultAzureCredential())

client = AzureOpenAI(
    api_key= os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

blobs_list = container_client.list_blobs(name_starts_with='customer_id_88_csvs/customer_id_88_jan_jun')

assistant = client.beta.assistants.create(
    instructions="""You are an expert auditor, skilled at categorizing transactions in the following categories: 
    income (Salary, bonuses, caskbacks etc.), Investments, Utilities, Bills, Transport, Fuel/Gas/Auto, Food (groceries, dining etc.), Shopping, Medical, Insurance, Entertainment, Travel, Education, Card Bills, Misc. 
    Note: Strictly only reply with one word - the respective category of transaction""",
    model="RG210-openai-4o",  
    tools=[]
    )

thread = client.beta.threads.create()

def categorize_transaction(description):
    # can be finetuned
    response = client.chat.completions.create(model='RG210-openai-35turbo', 
                                         max_tokens=20, 
                                         temperature=0,
                                         messages=[
                                             {
                                                 "role": "system",
                                                 "content": """You are an expert auditor, skilled at categorizing transactions in the following categories: 
                                                 Income (Salary, bonuses, caskbacks etc.), Investments, Utilities, Bills, Transport, Fuel/Gas/Auto, Food (groceries, dining etc.), Shopping, Medical, Insurance, Entertainment, Travel, Education, Card Bills, Misc.
                                                 The transactions can have credits (marked as 'CR') or debits (marked as 'DR'). 'CR' can not be categorized into outgoing balance categories, and DR can not be categorized into incoming categories.
                                                 Note: Strictly only reply with one word - the respective category of transaction."""
                                             }, {
                                                 "role": "user",
                                                 "content": description
                                             }
                                         ]
                                         )
    return response.choices[0].message.content.strip()

def preprocess_df(df):
    # will directly be DB
    df = df.rename(columns={'0': 'date', '1': 'description', '2': 'amount', '3': 'type'})
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df['amount'] = df['amount'].replace(',', '').astype(float)
    df['description'] = df['description'].str.lower()

    return df

dataframes = []
for blob in blobs_list:
    blob_client = container_client.get_blob_client(blob.name)
    download_stream = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(download_stream))
    dataframes.append(df)

dataframes = [preprocess_df(df) for df in dataframes]

for df in dataframes:
    df['category'] = df.apply(lambda row: categorize_transaction(' '.join(row.astype(str))), axis=1)

