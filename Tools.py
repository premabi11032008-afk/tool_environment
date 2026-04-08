from tavily import TavilyClient
import requests
import os

def websearch(search_topic:str):
    """
    this takes search the search_topic as the input return a json object that include all the search result
    """

    client=TavilyClient(api_key=os.getenv("TALVY_API_KEY"))
    return client.search(search_topic,search_depth="ultra-fast",max_results=3)["results"]

def get_weather(city:str):
    """
    this takes a city as a input and return the response about the weather in the city
    """

    data=requests.get(f"http://api.weatherapi.com/v1/current.json?key={os.getenv('WEATHER_API_KEY')}&q={city}&aqi=no")
    return data.json()["current"]

#print(get_weather("chennai"))
#print("-"*20)
#print(websearch("hi meaning"))
#print("-"*20)
#print(execute_sql_query("show databases"))
