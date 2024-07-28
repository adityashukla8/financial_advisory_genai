import json
import os

import requests
from langchain.tools import tool

from dotenv import load_dotenv
load_dotenv()

from ipdb import set_trace as ipdb

class SearchTool():
    @tool("Search internet")
    def search_internet(query):
        """search internet for the given query"""

        print("searching internet...")
        top_result_to_return = 5
        url = "https://google.serper.dev/search"
        payload = json.dumps(
            {"q": query, "num": top_result_to_return, "tbm": "nws"}
        )
        headers = {
            'X-API-KEY': os.environ['SERPER_API_KEY'],
            'content-type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        # ipdb()

        if 'organic' not in response.json():
            return "Sorry, couldn't find anything."
        else:
            results = response.json()['organic']
            string = []
            print('Results: ', results[:top_result_to_return])

            for result in results[:top_result_to_return]:
                try:
                    date = result.get('date', 'date not found')
                    string.append('\n'.join([
                        f"Title: {result['title']}",
                        f"Link: {result['link']}",
                        f"Date: {result['date']}",
                        f"Snippet: {result['snipper']}",
                        "\n--------------"
                    ]))
                except KeyError:
                    next

            return '\n'.join(string)
