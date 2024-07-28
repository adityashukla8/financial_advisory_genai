from crewai import Crew, Process

from langchain_openai import ChatOpenAI

from agents import FinNewsAgent
from tasks import AINewsTasks

OpenAIGPT4 = ChatOpenAI(model='gpt-4')

agents = FinNewsAgent()
tasks = AINewsTasks()

news_fetcher_agent = agents.news_fetcher_agent()
news_analyser_agent = agents.news_analyser_agent()
news_compiler_agent = agents.news_compiler_agent()

fetch_news_task = tasks.fetch_news_tasks(news_fetcher_agent)
analyze_news_task = tasks.analyze_news_task(news_analyser_agent, [fetch_news_task])

crew = Crew(
    agents = [news_fetcher_agent, news_analyser_agent, news_compiler_agent],
    tasks = [fetch_news_task, analyze_news_task],
    process = Process.hierarchical,
    manager_llm = OpenAIGPT4,
)

results = crew.kickoff()
print('Crew results: ')
print(results)
