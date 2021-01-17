from flask import Flask
import json
import requests
import tensorflow as tf
import numpy as np
import praw
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

def MLInferences(ListOfHeadlines,Vectorizer):

    ListOfHeadlines = Vectorizer.transform(ListOfHeadlines).toarray().tolist()
    
    interpreter = tf.lite.Interpreter(model_path="MainMode.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on random input data.
    input_shape = input_details[0]['shape']

    ReliabilityLabel = []
    for i in range(len(ListOfHeadlines)):
      input_data = np.array(ListOfHeadlines[i:i+1], dtype=np.float32)
      interpreter.set_tensor(input_details[0]['index'], input_data)

      interpreter.invoke()

      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor.
      output_data = interpreter.get_tensor(output_details[0]['index'])
      Result = output_data[0][0]
      if Result >= 0.5:
        ReliabilityLabel.append("True")
      else:
        ReliabilityLabel.append("False")

    return ReliabilityLabel


def ConvertRawPythonTitlesToList(RedditObj,Subreddit="worldnews",Limit=30):
    ResultList = []
    for Headline in RedditObj.subreddit(Subreddit).hot(limit=Limit):
        ResultList.append(Headline.title)
  
    return ResultList
 
 
def GetVectorizer():
    return pickle.load(open("Vectorizer.pickle","rb"))
 
 
 
@app.route("/")
def homepage():
    # Read in API key for the news API
    with open('RedditCreds.json') as JSONFile:
        RawJSON = json.load(JSONFile)
     
    ClientID =  RawJSON["ClientID"]
    ClientSecret =  RawJSON["ClientSecret"]
    UserAgent =  RawJSON["UserAgent"]
    
    RedditObj = praw.Reddit(
         client_id= ClientID,
         client_secret= ClientSecret ,
         user_agent=UserAgent
     )
    
    
    RawHeadlines = ConvertRawPythonTitlesToList(RedditObj)
    Vectorizer = GetVectorizer()
    FakeRealLabels = MLInferences(RawHeadlines, Vectorizer)
    
    ResultHTML = "<div style='text-align:center';font-family:'arial'>\n"
    for HeadlineNum in range(len(RawHeadlines)):
        ResultHTML += "<h3>{} - {}</h3>\n".format(RawHeadlines[HeadlineNum],FakeRealLabels[HeadlineNum])
       
    ResultHTML += "</div>\n"
    return ResultHTML



if __name__ == "__main__":
    app.run(debug=True)
