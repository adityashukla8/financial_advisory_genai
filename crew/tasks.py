from crewai import Task
from datetime import datetime

class AINewsTasks():
    def fetch_news_tasks(self, agent):
        return Task(
            description = 'Fetch latest top news in the investment/Indian finance space in the last 24 hours. The current time is {datetime.now()}',
            agent = agent,
            # async_execution = True,
            expected_output = """A list of top news in the Indian finance space - title, URLs, brief summary for each from last 24 hours. Example output:
            [
            {"title": "News Article Title",
            "url": "https://www.example.com",
            "summary": "Summary of the post",
            },
            {{...}}
            ]
            """
        )

    def analyze_news_task(self, agent, context):
        return Task(
            description = 'Analyze each news article to help deciding the investment decision.',
            agent = agent,
            # async_execution = True,
            context = context,
            expected_output = """Analysis of each news article in a well formatted manner.
            Expected output:
            Analysis of the subject
            """
        ) 
