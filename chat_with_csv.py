from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
from langchain_openai import AzureOpenAI

import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv()

llm = AzureOpenAI(
    deployment_name='RG210-openai-35turbo',
    api_version='2023-12-01-preview'
)

# can implement natural language to sql to query database
agent = create_csv_agent(
    # llm,
    OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0, model='gpt-3.5-turbo-instruct'),
    'artifacts/monthly_analysis.csv',
    verbose=True,
    allow_dangerous_code=True)

cust_data = pd.read_csv(r'artifacts/cat.csv')

# monthly_breakdown = agent.run('For each month, what is monthly spend in each category? Share in JSON format.')

