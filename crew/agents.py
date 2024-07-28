from crewai import Agent
from tools.search_tools import SearchTool

class FinNewsAgent():
    def news_fetcher_agent(self):
        return Agent(
            role = 'NewsFetcher',
            goal = 'Fetch latest news in Indian financial space',
            backstory = 'You are a passionate and skilled financial/investment analyst. Find latest relevant articles to help make investment decisions.',
            tools=[SearchTool.search_internet],
            verbose = True,
            allow_delegation = True,
            max_iter = 5
        )

    def news_analyser_agent(self):
        return Agent(
            role = 'NewsAnalyzer',
            goal = 'Analyze the news articles and create a conclusion summary on the subject.',
            backstory = 'You are a skilled investor who understands market movements and sentiments. Based on the article, have a conclusion about the subject.',
            tools=[SearchTool.search_internet],
            verbose = True,
            allow_delegation = True
        )

    def news_compiler_agent(self):
        return Agent(
            role = 'NewsCompiler',
            goal = 'Compile the analyzed news artiles to have a final conclusion on the subject.',
            backstory = 'Give a final verdict to the compiled news articles on the subject. The final conclusion you give will be used for investment decision.',
            verbose = True,
        )
