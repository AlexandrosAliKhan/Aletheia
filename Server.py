from flask import Flask # Make sure you have Flask
import json
import requests
import tensorflow as tf # Make sure you have tensorflow
import numpy as np # Make sure you have numpy
import praw # pip3 install praw
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer # make sure you have sklearn


app = Flask(__name__)

"""
To Do List:

- Use the templating engine to dynamically create good looking website (add Javascript,HTML,and CSS)
- Make a form(look into how post requests are handled) -> allows user to specify which subreddit to check for fake news
- Add any feature you like :) => Also add a Sentiment analysis icon/information about each post, pattern/keyword matching(filtering), link to each post
! NOT RECOMMENDED -> User accounts -> you will need to do OAUTH authentication for each login (not fun unless PRAW makes it easy)
! Do not add keys to GitHub page
- Deploy the website (Python-anywhere)


"""



def MLInferences(ListOfHeadlines,Vectorizer):
    """
    Input: Python list of headlines(strings), Pre-trained vectorizer from sklearn
    Output: A Python list of true/fake news labels for the headlines you have passed in (same order)
    """
    
    # Setu vectorizer
    ListOfHeadlines = Vectorizer.transform(ListOfHeadlines).toarray().tolist()
    
    # Set up interperter
    interpreter = tf.lite.Interpreter(model_path="MainMode.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    
    # Repeatly call the interperator can give it single value to inference
    ReliabilityLabel = []
    for i in range(len(ListOfHeadlines)):
      input_data = np.array(ListOfHeadlines[i:i+1], dtype=np.float32)
      interpreter.set_tensor(input_details[0]['index'], input_data)

      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])
      Result = output_data[0][0]
      
      # Give inferences string labels
      if Result >= 0.5:
        ReliabilityLabel.append("True")
      else:
        ReliabilityLabel.append("False")
    
    # Return labels for all the headlines(in the same order of the given headline list)
    return ReliabilityLabel


def ConvertRawPythonTitlesToList(RedditObj,Subreddit="worldnews",Limit=30):
    """
    Input: Reddit praw object, name of the subreddit you want to examine, and the number of posts you want to save
    Output: A Python list of all the headlines from the subreddit you have passed in.
    """
    ResultList = []
    for Headline in RedditObj.subreddit(Subreddit).hot(limit=Limit):
        ResultList.append(Headline.title)
  
    return ResultList
 
 
def GetVectorizer():
    # Make vectorizer useable in our program
    return pickle.load(open("Vectorizer.pickle","rb"))
 
 
 
@app.route("/")
def homepage():
    # Read in creditentials from the JSON file
    with open('RedditCreds/RedditCreds.json') as JSONFile:
        RawJSON = json.load(JSONFile)
     
    ClientID =  RawJSON["ClientID"]
    ClientSecret =  RawJSON["ClientSecret"]
    UserAgent =  RawJSON["UserAgent"]
    
    # Create the PRAW object you will use to interface with Reddit
    RedditObj = praw.Reddit(
         client_id= ClientID,
         client_secret= ClientSecret,
         user_agent=UserAgent
     )
    
    
    # Get a python list of titles from the desired subreddit
    RawHeadlines = ConvertRawPythonTitlesToList(RedditObj)
    
    # Get a usable vectorizer to pass the inferencing/prediction stage
    Vectorizer = GetVectorizer()
    
    # Get the list of fake/real news classifcation for the headlines you pass in
    FakeRealLabels = MLInferences(RawHeadlines, Vectorizer)
    
    # Prepare a temporary HTML document to give to client
    ResultHTML = "<div style='text-align:center';font-family:'arial'>\n"
    for HeadlineNum in range(len(RawHeadlines)):
        ResultHTML += "<h3>{} - {}</h3>\n".format(RawHeadlines[HeadlineNum],FakeRealLabels[HeadlineNum])
       
    ResultHTML += "</div>\n"
    
    return ResultHTML



if __name__ == "__main__":
    # When deploying -> change to production mode (python-anywhere)
    app.run(debug=True)
