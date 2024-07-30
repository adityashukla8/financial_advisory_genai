# {{{ imports

from fin_data_qa import analyze_financial_data
from vector_db_qa import query_vector_db

from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import yfinance as yf

from pydantic import BaseModel, Field
from typing import List

import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

from ipdb import set_trace as ipdb
# }}} 
# {{{ class: financial data analysis 

class FinancialDataAnalysisInput(BaseModel):
    """input for Financial data analysis"""

    query: str = Field(..., description = 'query string for financial data analysis')

class FinancialDataAnalysisTool(BaseTool):
    name = 'analyze_financial_data'
    description = """useful when you need to look up financial data for the user, their spending habits, retrieve their balance, look up their financial goals and risk profile, visualize data etc. Useful for any task that requires user's personal data.
    Note: Always provide concise, accurate and invormative, well formatted response, in bullets whenever possible."""

    def _run(self, query: str):
        with st.spinner('Looking up your financial data...'):
            fin_data_analysis_response = analyze_financial_data(query)
            st.success('✅ Analyzed your financial data')

        return fin_data_analysis_response

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = FinancialDataAnalysisInput
# }}} 
# {{{ class: query vector db

class RecommendProductsInput(BaseModel):
    """input for Recommend product by querying vector database"""

    query: str = Field(..., description = 'query striing to find relevant products from vector database')

class RecommendProductsTool(BaseTool):
    name = 'query_vector_db'
    description = """Useful when you want to recommend products or investment options or answer any Bank of Baroda specific questions.
    Note: always provide accurate and invormative, well formatted response."""

    def _run(self, query: str):
        with st.spinner('Looking up BoB database...'):
            query_vector_db_response = query_vector_db(query)
            st.success('✅ Fetched relevant info from BoB database')

        return query_vector_db_response

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = RecommendProductsInput

# }}} 
# {{{ def: yahoo fin api

def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    try:
        todays_data = ticker.history(period='1d')
        return round(todays_data['Close'][0], 2)
    except:
        return 0
# }}} 
# {{{ class: Stock price lookup tool 

class StockPriceLookupInput(BaseModel):
    """input for Stock price look-up tool"""

    symbol: str = Field(..., description='ticker symbol to look up stocks price')

class StockPriceLookupTool(BaseTool):
    name = 'get_stock_price'
    description = 'useful when you want to look up price of a specific company using its ticker/symbol'

    def _run(self, symbol: str):
        with st.spinner('Looking up stock price...'):
            stock_price_response = get_stock_price(symbol)
            st.success('✅ Looked up stock price')

        return stock_price_response

    def _arun(self, symbol: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceLookupInput
# }}} 
# {{{ def: google search

tavily_tool = TavilySearchResults(max_results=20)
def search_investment_options(search_query):
    search_response = tavily_tool.invoke(search_query)
    
    return search_response
# }}} 
# {{{ class: search investment options/stocks

class SearchInvestmentOptionsInput(BaseModel):
    """input for Search Investment Options/Stocks"""
    search_query: str = Field(..., description = "search query to search google for investment options/stocks based on user's risk profile and investment goals")

class SearchInvestmentOptionsTool(BaseTool):
    name = 'search_investment_options'
    description = """useful when you want to search google for finding investment options/list of good stocks, based on user's risk profile and investment goals. 
    Note:
    - Always reformat the input query in a way that will give the best search results.
    - Search strictly for India specific. 
    - In case of stocks, return stock's actual ticker/symbol. 
    - You can access user's risk profile and goals by calling 'analyze_financial_data'
    - Attempt to make the responses Bank of Baroda centric if possible.
    - Post-search, always call 'query_vector_db' to search for relevant products and also recommend them along the answer.
    """
    
    def _run(self, search_query: str):
        with st.spinner('Searching google for investment options...'):
            search_response = search_investment_options(search_query)
            st.success('✅ Searched google for investment options')
        return search_response

    def _arun(self, search_query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = SearchInvestmentOptionsInput

# }}} 

# {{{ initialize tools, llm and openai agent 

tools = [FinancialDataAnalysisTool(), RecommendProductsTool(), StockPriceLookupTool(), SearchInvestmentOptionsTool()]

llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

open_ai_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
# open_ai_agent.run('what is my spending habit')
# }}} 
# {{{ streamlit config 

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title='BoB Bot', page_icon='🏆')
st.title('BoB Bot - Financial Advisory')

# get response
def get_response(query, chat_history):
    template = """You are an expert financial advisor at Bank of Baroda, assisting users with informative and accurate answers.

    Chat history: {chat_history}
    User question: {user_question}"""

    prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo-0125')

    chain = prompt | open_ai_agent
    with st.spinner('Processing...'):
        result = chain.invoke({"chat_history": chat_history, "user_question": query})
    # result = chain.invoke(query)
    # ipdb()
    return result['output']
    # return chain.invoke({
    #     "chat_history": chat_history,
    #     "user_question": query
    # })

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.markdown(message.content)
    else:
        with st.chat_message('AI'):
            st.markdown(message.content)

user_query = st.chat_input("Your message...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))

# }}} 
#TODO: test for indian stocks
