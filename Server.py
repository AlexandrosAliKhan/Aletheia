from flask import Flask # Make sure you have Flask
from flask import render_template
import json
import requests
import tensorflow as tf # Make sure you have tensorflow
import numpy as np # Make sure you have numpy
import praw # pip3 install praw
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer # make sure you have sklearn
from flask import request
from textblob import TextBlob
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

def SentimentAnalysis(ListOfHeadlines):
    SentimentListPolarity = []
    SentimentListSubjectivity = []
    # Parse headlines and make lists for polarity and subjectivity for each headline(in order)
    for Headline in ListOfHeadlines:
        TextBlobObjHeadline = TextBlob(Headline)
        SentimentListPolarity.append(TextBlobObjHeadline.sentiment.polarity)
        SentimentListSubjectivity.append(TextBlobObjHeadline.sentiment.subjectivity)
    
    return SentimentListPolarity, SentimentListSubjectivity

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
        ReliabilityLabel.append("Fake")
    
    # Return labels for all the headlines(in the same order of the given headline list)
    return ReliabilityLabel


def ConvertRawPythonTitlesToLists(RedditObj,Subreddit=None,cat="hot",Limit=30):
    """
    Input: Reddit praw object, name of the subreddit you want to examine, and the number of posts you want to save
    Output: A Python list of all the headlines from the subreddit you have passed in.
    """
    ResultList = []
    linklist = []
    if Subreddit is None:
        if cat == "hot":
            for Headline in RedditObj.front.hot(limit=Limit):
               ResultList.append(Headline.title)
               linklist.append(Headline.url)
            return ResultList, linklist
        elif cat == "new":
            for Headline in RedditObj.front.new(limit=Limit):
               ResultList.append(Headline.title)
               linklist.append(Headline.url)
            return ResultList, linklist
        elif cat == "top":
            for Headline in RedditObj.front.top(limit=Limit):
               ResultList.append(Headline.title)
               linklist.append(Headline.url)
            return ResultList, linklist
        elif cat == "rising":
            for Headline in RedditObj.front.rising(limit=Limit):
               ResultList.append(Headline.title)
               linklist.append(Headline.url)
            return ResultList, linklist
    else:
        if cat == "hot":
            for Headline in RedditObj.subreddit(Subreddit).hot(limit=Limit):
               ResultList.append(Headline.title)
               linklist.append(Headline.url)
            return ResultList, linklist
        elif cat == "new":
            for Headline in RedditObj.subreddit(Subreddit).new(limit=Limit):
               ResultList.append(Headline.title)
               linklist.append(Headline.url)
            return ResultList, linklist
        elif cat == "top":
            for Headline in RedditObj.subreddit(Subreddit).top(limit=Limit):
               ResultList.append(Headline.title)
               linklist.append(Headline.url)
            return ResultList, linklist
 
 
def GetVectorizer():
    # Make vectorizer useable in our program
    return pickle.load(open("Vectorizer.pickle","rb"))
 
 
 
@app.route("/")
@app.route("/<cat>/")
def homepage(cat="hot",q=None):
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
    if q is None:
        RawHeadlines, links = ConvertRawPythonTitlesToLists(RedditObj, cat=cat)
    else:
        RawHeadlines, links = ConvertRawPythonTitlesToLists(RedditObj, q, cat=cat)
    
    # Get a usable vectorizer to pass the inferencing/prediction stage
    Vectorizer = GetVectorizer()
    # Get lists from sentiment analysis(polarity and subjectivity)
    SentimentListPolarity, SentimentListSubjectivity = SentimentAnalysis(RawHeadlines)
    # Get the list of fake/real news classifcation for the headlines you pass in
    FakeRealLabels = MLInferences(RawHeadlines, Vectorizer)
    
    '''
    # Prepare a temporary HTML document to give to client
    ResultHTML = "<div style='text-align:center';font-family:'arial'>\n"
    for HeadlineNum in range(len(RawHeadlines)):
        ResultHTML += "<h3>{} - {}</h3>\n".format(RawHeadlines[HeadlineNum],FakeRealLabels[HeadlineNum])
        
    ResultHTML += "</div>\n"
    
    return ResultHTML
    '''

    return render_template('index.html', rhels=RawHeadlines, pabels=FakeRealLabels, links=links, cat=cat, q=q, polars=SentimentListPolarity, sentis=SentimentListSubjectivity)

@app.route("/search/", methods=['GET', 'POST'])
def search():
	subreddit = request.args.get('q')
	print(subreddit)
	return homepage("hot",subreddit)


if __name__ == "__main__":
    # When deploying -> change to production mode (python-anywhere)
    app.run(debug=True)

