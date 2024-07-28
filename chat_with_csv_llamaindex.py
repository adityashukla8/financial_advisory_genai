# {{{ imports 

import logging
import sys
from IPython.display import Markdown, display
import os
from dotenv import load_dotenv

import pandas as pd

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core.query_pipeline import (
    QueryPipeline as QP, Link, InputComponent
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core import PromptTemplate

from pyvis.network import Network

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# }}} 
# {{{ env variables

AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# }}} 
# {{{ initialize llm 

azure_llm = AzureOpenAI(
    model = 'gpt-35-turbo-16k',
    deployment_name = 'RG210-openai-35turbo',
    api_key = api_key,
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    # api_version = ,
    temperature = 0.1
)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
# }}} 
# {{{ load data 

cust_data = pd.read_csv(r'./artifacts/cat.csv')
user_data = pd.read_csv(r'./artifacts/user_info.csv')
user_data_dict = user_data.to_dict(orient='records')
# }}} 
# query_engine = PandasQueryEngine(df=cust_data, verbose=True)
# response = query_engine.query('what is the data about')
# {{{ prompts

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

# user_data_response_synthesis_prompt_str = (
#     "You are a skilled financial advisor, expert at deriving financial insights from user information, risk capacity and other factors.\n"
#     ""
# )

response_synthesis_prompt_str = (
    "You are a highly experienced Indian financial advisor, skilled are analysing customer financial data and recommending informative insights for investment.\n"
    "This is the customer information:\n"
    "{user_data_dict}"
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=cust_data.head(5),
)
pandas_prompt_user_data = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=user_data.head(5)
)

pandas_output_parser = PandasInstructionParser(cust_data)
# pandas_output_parser_user_data = PandasInstructionParser(user_data)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str).partial_format(user_data_dict=user_data_dict)
# }}} 
# {{{ query pipeline 

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        # "pandas_prompt2": pandas_prompt_user_data,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)

qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
        Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output",),
    ]
)
qp.add_link("response_synthesis_prompt", "llm2")
# }}} 
# {{{ visualize dag 

net = Network(notebook=True, cdn_resources='in_line', directed=True)
net.from_nx(qp.dag)
net.save_graph('query_pipeline_dag.html')
# }}} 
# {{{ run pipeline

def get_financial_insights(query):
    response = qp.run(query_str = query)
    return response.message.content
# }}}
