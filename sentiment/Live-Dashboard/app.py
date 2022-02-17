#Imports
from bs4 import BeautifulSoup
import pandas as pd # data manipulatio
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
from MCForecastTools import MCSimulation
import plotly.graph_objects as go
import panel as pn

pn.extension('plotly','tabulator' , 'echarts')

from PIL import Image
from html2image import Html2Image
import param 
import warnings
warnings.filterwarnings('ignore')
# loading the reqed libraries for scraping from Twitter and Reddit 
import tweepy # twitter api module 
import plotly.express as px 
import os
import hvplot.pandas # plotting and visuliaizng the data 
import praw # reddit api module 
import os  # os module to access the .env file 
from dotenv import load_dotenv 
import torch # required library for transformers models 

from bs4 import BeautifulSoup as bs # scraping and html parsing for some sites
import requests as rq # required for scraping from finviz.com
from transformers import AutoTokenizer, AutoModelForSequenceClassification # NLP 
import json # used to parse the http request data from finviz or cryptopanic
# easier way to scrape from finviz.com 
import finviz
import re
from PIL import Image
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from plotly.subplots import make_subplots
import matplotlib as mpl
from lxml import etree

text_field = pn.widgets.TextInput(name='Asset', value="AAPL" , placeholder = "enter Asset name ")
toggle = pn.widgets.ToggleGroup(options=['Stocks', 'Crypto'], behavior='radio', button_type="default")
no_tweets = pn.widgets.TextInput(name='Number of Tweets', value="100" , placeholder = "enter Asset name " , max_length =3 )
button = pn.widgets.Button(name='Submit')
#info = pn.widgets.StaticText(value=df_metrics_long['value'][3])

load_dotenv()
AlphaVantageKey = os.getenv("AlphaVantageKey")


url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=' + str(text_field.param.value) +'&apikey=' + str(AlphaVantageKey)
r = rq.get(url)
stock_metrics = r.json()
df_metrics = pd.DataFrame(stock_metrics,index = range(1))
df_metrics_long = df_metrics.melt()
metric_table = pn.widgets.Tabulator(df_metrics_long, width=500)
#info = pn.widgets.StaticText(value=df_metrics_long['value'][3])
@pn.depends(text_field.param.value,no_tweets.param.value)
def tabs(text_field,no_tweets):
    
    """Twitter Sentiment Data Analysis"""
    
    #create the twitter api access varibles 
    twit_api= os.getenv("TWIT_API")
    twit_secret = os.getenv("TWIT_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    # authentication for the twitter api 
    auth = tweepy.OAuthHandler(twit_api, twit_secret)
    auth.set_access_token(access_token, access_token_secret)
    #createing the API object
    t_api = tweepy.API(auth)
    #createing the API object
    t_api = tweepy.API(auth)
    # search tweets
    keyword = text_field
    tweets = tweepy.Cursor(t_api.search_tweets, q=keyword, count=int(no_tweets), tweet_mode='extended').items(int(no_tweets))
    # create DataFrame
    columns = ['ID','User', 'Tweet' , 'Time']
    data = []
    for tweet in tweets:
        data.append([tweet.id , tweet.user.screen_name, tweet.full_text , tweet.user.created_at])
        # clean the tweets from the RT and and @user calls using the lambda function in a two step process

    tweet_list = pd.DataFrame(data, columns=columns)
    
    pattern = '(RT @)\w+:'
    tweet_list['Tweet'] =tweet_list['Tweet'].apply(lambda x: re.sub(pattern , '' ,x))
    pattern = '(@)\w+'
    tweet_list['Tweet'] =tweet_list['Tweet'].apply(lambda x: re.sub(pattern , '' ,x))
        # create tokenizer variable to chose the model 
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        # encode the list of tweet to create the sentiment score from a function
    def sentiment_score(tweet):
        tokens = tokenizer.encode(tweet, return_tensors='pt')
        result = model(tokens)
        return int(torch.argmax(result.logits))+1
    tweet_list['sentiment'] = tweet_list['Tweet'].apply(lambda x: sentiment_score(x[:]))
        # do some data manipulation to work out some cool statistics and visuals 
    tweet_list.drop_duplicates(inplace=True)
    tweet_list.dropna(inplace=True)
    # add year , month and day to the dataset
    tweet_list['year']= tweet_list['Time'].dt.year
    tweet_list['month']= tweet_list['Time'].dt.month
    tweet_list['day']= tweet_list['Time'].dt.day
    #set the index to those columns 
    tweet_list.set_index(['year','month','day'] , inplace=True)
       
       
    #create a count of sentiment and persentage for pie charting
    sentiment_count = tweet_list['sentiment'].value_counts()
    #sentiment_count.to_frame()
    sentiment_count.sort_index(inplace=True)
    
    #get 10 positive and negative tweets 
    largest = tweet_list.nlargest(10, columns='sentiment').reset_index()
    pd.to_datetime(largest['Time'] , format='%d/%m/%y')
    largest_sen_tweets = largest[['Time','User','Tweet' , 'sentiment']]
    smallest = tweet_list.nsmallest(10, columns = 'sentiment').reset_index()
    pd.to_datetime(smallest['Time'] , format='%d/%m/%y')
    smallest_sen_tweets = smallest[['Time','User','Tweet','sentiment']] 

    """
    total = sentiment_count.sum()
    sentiment_count.get(1)
    sentiment_count.get(2)
    sentiment_count.get(3)
    negative_senti =round((sentiment_count[1]/total)*100,2)
    neutral_senti =round((sentiment_count[2]/total)*100,2)
    positive_senti =round((sentiment_count[3]/total)*100,2)
        """
    #Create a pie chart and store the object in pie_pane
    fig0 = Figure(figsize=(4,4))
    ax0 = fig0.subplots()
    pie = ax0.pie(sentiment_count,
                  labels=['Negative','Neutral','Positive'],
                  autopct = '%.2f',
                  shadow=True,
                  startangle=90)
    
    pie_pane = pn.pane.Matplotlib(fig0, dpi=144 , tight = True)
    
    #group_tweet_senti = tweet_list.groupby(by='sentiment').count()
    #group_tweet_senti.drop(columns=['Time','User'] ,inplace=True)
    
    #plot = group_tweet_senti.hvplot.bar(x='sentiment')
    #plot2 = group_tweet_senti.hvplot.hist(y='User')
    #plot3= group_tweet_senti.hvplot.hist(y='Tweet')
    
    #Hist_plots = pn.Column(plot,plot2,plot3)
    
    """Consumer Sentiment Word Plot results."""


    # CODE HERE
    text = ' '.join(tweet_list['Tweet'])

    comment_words = ""
    stopwords = set(STOPWORDS)
    #stopwords = set(STOPWORDS)
    custom_stop_words = ["https" , 't', ]
    [stopwords.add(n) for n in custom_stop_words]
    
    wc = WordCloud(background_color="white" , width=500, max_words=3500, 
              stopwords=stopwords, max_font_size=100, random_state=42)

    fig = plt.figure()
    mpl.rcParams["figure.figsize"]=[10.0,5.0]
    #generate word cloud
    wc_image= wc.generate(text)
    plt.imshow(wc_image)
    plt.axis("off")
    fontdict={"fontsize": 48, "fontweight":"bold"}
    plt.title(f"{text_field} Word Cloud", fontdict=fontdict)
    plt.close(fig)
    
    """Get Stock price data and tabulate metrics"""
    
    # Read API keys into env 
    # Read the API keys
    load_dotenv()
    AlphaVantageKey = os.getenv("AlphaVantageKey")
    
    # Set Alpaca API key and secret
    alpaca_api_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    api = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        api_version = "v2"
    )
    
    #Get stock data
    # Set timeframe to '1D'
    timeframe = "1D"
    
    # Set start and end datetimes between now and 3 years ago.
    start_date = pd.Timestamp("2018-05-01", tz="America/New_York").isoformat()
    end_date = pd.Timestamp("2022-01-19", tz="America/New_York").isoformat()
    
    # Set the ticker information
    tickers = text_field
    
    # Get 3 year's worth of historical price data
    
    df_ticker = api.get_barset(
        tickers,
        timeframe,
        start=start_date,
        end=end_date,
        limit=1000,
    ).df
    
    
    #initilize some varibles 
    ema = True
    periods = 14
    close_delta = df_ticker.iloc[:,3].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    df_ticker['RSI'] = rsi
    df_ticker['RSI'].dropna().head(20)
    df_ticker['RSI UCL'] = 70
    df_ticker['RSI LCL'] = 30
    
    #rsi_plot = px.line(x=df_ticker.index,y = df_ticker.RSI, title='RSI Chart')
    #rsi_plot.add_scatter(x=df_ticker.index,y = df_ticker['RSI UCL'],name='BUY')
    #rsi_plot.add_scatter(x=df_ticker.index,y = df_ticker['RSI LCL'],name='SELL')
    
    # calculate Bolinger Bands 
    df_ticker['SMA']=df_ticker.iloc[:,3].rolling(window=13).mean()
    bolly_band_std=df_ticker.iloc[:,3].rolling(window=13).std()
    df_ticker['UB']=df_ticker['SMA']+(bolly_band_std*2)
    df_ticker['LB']=df_ticker['SMA']-(bolly_band_std*2)
    
    df_ticker.dropna(inplace=True)
    
    #create candelstick chart 
    url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=' + str(tickers) +'&apikey=' + str(AlphaVantageKey)
    r = rq.get(url)
    stock_metrics = r.json()
    df_metrics = pd.DataFrame(stock_metrics,index = range(1))
    df_metrics_long = df_metrics.melt()
    metric_table = pn.widgets.Tabulator(df_metrics_long, width=1000)
    
    final_fig = make_subplots(rows=2 , cols =1)
    
    final_fig.append_trace(go.Candlestick(
                                            x=df_ticker.index,
                                            open=df_ticker[tickers,"open"],
                                            high=df_ticker[tickers,"high"],
                                            low=df_ticker[tickers,"low"],
                                            close=df_ticker[tickers,"close"]
                                            
                                        ),
                         row = 1, col=1)
    
    final_fig.append_trace(go.Scatter(x=df_ticker.index , y = df_ticker['RSI']), row=2,col=1)
    
    
    final_fig.update_layout(title= f"Price Chart for {text_field}")
    final_fig.update_traces(name='Price', selector=dict(type='candlestick'))
    final_fig.update_traces(name='RSI', selector=dict(type='Scatter'))
    final_fig.update_layout(xaxis_rangeslider_visible=False)
    #final_fig.update_layout(uirevision=df_ticker)
    """
    fig = go.Figure(data=[go.Candlestick(
                                            x=df_ticker.index,
                                            open=df_ticker[tickers,"open"],
                                            high=df_ticker[tickers,"high"],
                                            low=df_ticker[tickers,"low"],
                                            close=df_ticker[tickers,"close"]
                                        )
                         ],
                    )
                   )
    fig.add_trace(go.Line(x=df_ticker.index,y = df_ticker['UB'],name='Upper Band'))
    fig.add_trace(go.Line(x=df_ticker.index,y = df_ticker['LB'],name='Lower Band'))

    #fig2=go.Figure(data=[go.Scatter(x=df_ticker.index , y = df_ticker['RSI'],name='RSI')])


    fig.add_trace(go.Line(x=df_ticker.index,y = df_ticker['RSI'],name='RSI'))
    """ 
    """ Creating the Simulation Plots """
    
    df_stock_data = df_ticker[tickers,"close"]
    stock_close_df = df_stock_data.to_frame()
    
    MC_thirtyyear = MCSimulation(
    portfolio_data = stock_close_df,
    weights = [1],
    num_simulation = 10,
    num_trading_days = 252*30)
    
    MC_thirtyyear.calc_cumulative_return()
    
    simulation_df, plot_title = MC_thirtyyear.plot_simulation()
    line_plot = simulation_df.hvplot(title=plot_title, legend=False).opts(yformatter="%.0f")
    
    """Fear & Greed Index results."""

    #hti = Html2Image()
    #url_data = 'https://money.cnn.com/data/fear-and-greed/'
    #hti.screenshot(url='https://money.cnn.com/data/fear-and-greed/', save_as='fear_reed.png')
    ## Importing Image class from PIL modul
    ## Opens a image in RGB mode
    #im = Image.open(r"fear_reed.png")
    ## Setting the points for cropped image
    #left = 470
    #top = 80
    #right = 1100
    #bottom = 400
    ## Cropped image of above dimension
    ## (It will not change original image)
    #im1 = im.crop((left, top, right, bottom))
    ## Shows the image in image viewer
    #im1.save("pic.png" , format="png")
    #fng_fig = pn.pane.PNG("pic.png",alt_text='F&G' , width = 600)

    #get the url to scrape from cnn
    url = 'https://money.cnn.com/data/fear-and-greed'
    res = rq.get(url)
    #create a soup object to parse the page
    soup = BeautifulSoup(res.content, 'html.parser')
    # create an etreee object to use Xapth attribute
    object_tree = etree.HTML(str(soup))

    # create and empty list to store the data being scraped
    list_fng = []

    # loop over the data and store in the list , this will grab the fear and greed tabulated data from the page
    for i in range(5):
        list_fng.append([object_tree.xpath('/html/body/div[3]/div[1]/div[1]/div[3]/div/div[1]')[0][0][i].text])

    df_fng = pd.DataFrame(list_fng)
    for i in df_fng:
        df_fng['Value'] = df_fng[i].str.extract(pat ='([0-9]{2})')
        df_fng['Status'] = df_fng[i].str.extract(pat= '((?<=\().+?(?=\)))')
        df_fng['Time'] = df_fng[i].str.replace(pat='(\:.*$)' , repl = '')

    df_fng.drop(columns=0 , inplace=True)

    fng_gauge = pn.indicators.Gauge(
        name='Fear and Greed Index',
        value=int(df_fng['Value'][0]),
        bounds=(0, 100), format='{value} %',
        colors=[(0.25, 'red'), (0.5, 'orange'), (0.75, 'yellow') , (1 , 'green')])


    """ Bot recomendation based on the metrics"""
    
    def bot():
        rsi = df_ticker['RSI']
        pe = df_metrics_long['value'][15]
        fear_greed = df_fng['Value'][0]
        """
        score = 0 
        
        if rsi > 70:
            score+=1
        elif rsi <30:
            score-=1
        else:
             score+=0           
        if pe > 30:
            score-=1
        elif pe>15:
            score+=1
        else:
            score+=0            
        if fear_greed > 75:
            score+=1
        elif fear_greed <30:
            score-=1
        else:
            score+=0 """
        #msg = f'The final score based on the {score} with the follwing paramaters used:\nRSI was {rsi}\nPERatio was {pe}\nFear and Greed Index was {fear_greed}'
        
        return "Still a work in progress"
            
        
    
    
    """ Layout Design of the dashboard"""
    # Layout design 
    twitter = pn.Row(pn.widgets.DataFrame(largest_sen_tweets,name='High Sentiment Tweets'),
                    pn.widgets.DataFrame(smallest_sen_tweets,  name='Low Sentiment Tweets'))

    fng_word = pn.Row(pie_pane , pn.pane.Matplotlib(fig) , fng_gauge)
    
    
    sentiment = pn.Column(fng_word ,twitter)
     
   
    Data_panel = pn.Tabs(
        ('Charts' , final_fig),
        ('Stock Metrics Data' , metric_table),
        ('Simulation' , line_plot),
        ('Sentiment' , sentiment),
        ('Recomendation' , bot()))
    
          
    return Data_panel

text_field = pn.widgets.TextInput(name='Asset', value="AAPL" , placeholder = "enter Asset name ")
toggle = pn.widgets.ToggleGroup(options=['Stocks', 'Crypto'], behavior='radio', button_type="default")
no_tweets = pn.widgets.TextInput(name='Number of Tweets', value="100" , placeholder = "enter Asset name " , max_length =3 )
button = pn.widgets.Button(name='Submit')
#info = pn.widgets.StaticText(value=df_metrics_long['value'][3])

load_dotenv()
AlphaVantageKey = os.getenv("AlphaVantageKey")


url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=' + str(text_field.param.value) +'&apikey=' + str(AlphaVantageKey)
r = rq.get(url)
stock_metrics = r.json()
df_metrics = pd.DataFrame(stock_metrics,index = range(1))
df_metrics_long = df_metrics.melt()
metric_table = pn.widgets.Tabulator(df_metrics_long, width=500)
#info = pn.widgets.StaticText(value=df_metrics_long['value'][3])
@pn.depends(text_field.param.value,no_tweets.param.value)
def tabs(text_field,no_tweets):
    
    """Twitter Sentiment Data Analysis"""
    
    #create the twitter api access varibles 
    twit_api= os.getenv("TWIT_API")
    twit_secret = os.getenv("TWIT_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    # authentication for the twitter api 
    auth = tweepy.OAuthHandler(twit_api, twit_secret)
    auth.set_access_token(access_token, access_token_secret)
    #createing the API object
    t_api = tweepy.API(auth)
    #createing the API object
    t_api = tweepy.API(auth)
    # search tweets
    keyword = text_field
    tweets = tweepy.Cursor(t_api.search_tweets, q=keyword, count=int(no_tweets), tweet_mode='extended').items(int(no_tweets))
    # create DataFrame
    columns = ['ID','User', 'Tweet' , 'Time']
    data = []
    for tweet in tweets:
        data.append([tweet.id , tweet.user.screen_name, tweet.full_text , tweet.user.created_at])
        # clean the tweets from the RT and and @user calls using the lambda function in a two step process

    tweet_list = pd.DataFrame(data, columns=columns)
    
    pattern = '(RT @)\w+:'
    tweet_list['Tweet'] =tweet_list['Tweet'].apply(lambda x: re.sub(pattern , '' ,x))
    pattern = '(@)\w+'
    tweet_list['Tweet'] =tweet_list['Tweet'].apply(lambda x: re.sub(pattern , '' ,x))
        # create tokenizer variable to chose the model 
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        # encode the list of tweet to create the sentiment score from a function
    def sentiment_score(tweet):
        tokens = tokenizer.encode(tweet, return_tensors='pt')
        result = model(tokens)
        return int(torch.argmax(result.logits))+1
    tweet_list['sentiment'] = tweet_list['Tweet'].apply(lambda x: sentiment_score(x[:]))
        # do some data manipulation to work out some cool statistics and visuals 
    tweet_list.drop_duplicates(inplace=True)
    tweet_list.dropna(inplace=True)
    # add year , month and day to the dataset
    tweet_list['year']= tweet_list['Time'].dt.year
    tweet_list['month']= tweet_list['Time'].dt.month
    tweet_list['day']= tweet_list['Time'].dt.day
    #set the index to those columns 
    tweet_list.set_index(['year','month','day'] , inplace=True)
       
       
    #create a count of sentiment and persentage for pie charting
    sentiment_count = tweet_list['sentiment'].value_counts()
    #sentiment_count.to_frame()
    sentiment_count.sort_index(inplace=True)
    
    #get 10 positive and negative tweets 
    largest = tweet_list.nlargest(10, columns='sentiment').reset_index()
    pd.to_datetime(largest['Time'] , format='%d/%m/%y')
    largest_sen_tweets = largest[['Time','User','Tweet' , 'sentiment']]
    smallest = tweet_list.nsmallest(10, columns = 'sentiment').reset_index()
    pd.to_datetime(smallest['Time'] , format='%d/%m/%y')
    smallest_sen_tweets = smallest[['Time','User','Tweet','sentiment']] 

    """
    total = sentiment_count.sum()
    sentiment_count.get(1)
    sentiment_count.get(2)
    sentiment_count.get(3)
    negative_senti =round((sentiment_count[1]/total)*100,2)
    neutral_senti =round((sentiment_count[2]/total)*100,2)
    positive_senti =round((sentiment_count[3]/total)*100,2)
        """
    #Create a pie chart and store the object in pie_pane
    fig0 = Figure(figsize=(4,4))
    ax0 = fig0.subplots()
    pie = ax0.pie(sentiment_count,
                  labels=['Negative','Neutral','Positive'],
                  autopct = '%.2f',
                  shadow=True,
                  startangle=90)
    
    pie_pane = pn.pane.Matplotlib(fig0, dpi=144 , tight = True)
    
    #group_tweet_senti = tweet_list.groupby(by='sentiment').count()
    #group_tweet_senti.drop(columns=['Time','User'] ,inplace=True)
    
    #plot = group_tweet_senti.hvplot.bar(x='sentiment')
    #plot2 = group_tweet_senti.hvplot.hist(y='User')
    #plot3= group_tweet_senti.hvplot.hist(y='Tweet')
    
    #Hist_plots = pn.Column(plot,plot2,plot3)
    
    """Consumer Sentiment Word Plot results."""


    # CODE HERE
    text = ' '.join(tweet_list['Tweet'])

    comment_words = ""
    stopwords = set(STOPWORDS)
    #stopwords = set(STOPWORDS)
    custom_stop_words = ["https" , 't', ]
    [stopwords.add(n) for n in custom_stop_words]
    
    wc = WordCloud(background_color="white" , width=500, max_words=3500, 
              stopwords=stopwords, max_font_size=100, random_state=42)

    fig = plt.figure()
    mpl.rcParams["figure.figsize"]=[10.0,5.0]
    #generate word cloud
    wc_image= wc.generate(text)
    plt.imshow(wc_image)
    plt.axis("off")
    fontdict={"fontsize": 48, "fontweight":"bold"}
    plt.title(f"{text_field} Word Cloud", fontdict=fontdict)
    plt.close(fig)
    
    """Get Stock price data and tabulate metrics"""
    
    # Read API keys into env 
    # Read the API keys
    load_dotenv()
    AlphaVantageKey = os.getenv("AlphaVantageKey")
    
    # Set Alpaca API key and secret
    alpaca_api_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    api = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        api_version = "v2"
    )
    
    #Get stock data
    # Set timeframe to '1D'
    timeframe = "1D"
    
    # Set start and end datetimes between now and 3 years ago.
    start_date = pd.Timestamp("2018-05-01", tz="America/New_York").isoformat()
    end_date = pd.Timestamp("2022-01-19", tz="America/New_York").isoformat()
    
    # Set the ticker information
    tickers = text_field
    
    # Get 3 year's worth of historical price data
    
    df_ticker = api.get_barset(
        tickers,
        timeframe,
        start=start_date,
        end=end_date,
        limit=1000,
    ).df
    
    
    #initilize some varibles 
    ema = True
    periods = 14
    close_delta = df_ticker.iloc[:,3].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    df_ticker['RSI'] = rsi
    df_ticker['RSI'].dropna().head(20)
    df_ticker['RSI UCL'] = 70
    df_ticker['RSI LCL'] = 30
    
    #rsi_plot = px.line(x=df_ticker.index,y = df_ticker.RSI, title='RSI Chart')
    #rsi_plot.add_scatter(x=df_ticker.index,y = df_ticker['RSI UCL'],name='BUY')
    #rsi_plot.add_scatter(x=df_ticker.index,y = df_ticker['RSI LCL'],name='SELL')
    
    # calculate Bolinger Bands 
    df_ticker['SMA']=df_ticker.iloc[:,3].rolling(window=13).mean()
    bolly_band_std=df_ticker.iloc[:,3].rolling(window=13).std()
    df_ticker['UB']=df_ticker['SMA']+(bolly_band_std*2)
    df_ticker['LB']=df_ticker['SMA']-(bolly_band_std*2)
    
    df_ticker.dropna(inplace=True)
    
    #create candelstick chart 
    url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=' + str(tickers) +'&apikey=' + str(AlphaVantageKey)
    r = rq.get(url)
    stock_metrics = r.json()
    df_metrics = pd.DataFrame(stock_metrics,index = range(1))
    df_metrics_long = df_metrics.melt()
    metric_table = pn.widgets.Tabulator(df_metrics_long, width=1000)
    
    final_fig = make_subplots(rows=2 , cols =1)
    
    final_fig.append_trace(go.Candlestick(
                                            x=df_ticker.index,
                                            open=df_ticker[tickers,"open"],
                                            high=df_ticker[tickers,"high"],
                                            low=df_ticker[tickers,"low"],
                                            close=df_ticker[tickers,"close"]
                                            
                                        ),
                         row = 1, col=1)
    
    final_fig.append_trace(go.Scatter(x=df_ticker.index , y = df_ticker['RSI']), row=2,col=1)
    
    
    final_fig.update_layout(title= f"Price Chart for {text_field}")
    final_fig.update_traces(name='Price', selector=dict(type='candlestick'))
    final_fig.update_traces(name='RSI', selector=dict(type='Scatter'))
    final_fig.update_layout(xaxis_rangeslider_visible=False)
    #final_fig.update_layout(uirevision=df_ticker)
    """
    fig = go.Figure(data=[go.Candlestick(
                                            x=df_ticker.index,
                                            open=df_ticker[tickers,"open"],
                                            high=df_ticker[tickers,"high"],
                                            low=df_ticker[tickers,"low"],
                                            close=df_ticker[tickers,"close"]
                                        )
                         ],
                    )
                   )
    fig.add_trace(go.Line(x=df_ticker.index,y = df_ticker['UB'],name='Upper Band'))
    fig.add_trace(go.Line(x=df_ticker.index,y = df_ticker['LB'],name='Lower Band'))

    #fig2=go.Figure(data=[go.Scatter(x=df_ticker.index , y = df_ticker['RSI'],name='RSI')])


    fig.add_trace(go.Line(x=df_ticker.index,y = df_ticker['RSI'],name='RSI'))
    """ 
    """ Creating the Simulation Plots """
    
    df_stock_data = df_ticker[tickers,"close"]
    stock_close_df = df_stock_data.to_frame()
    
    MC_thirtyyear = MCSimulation(
    portfolio_data = stock_close_df,
    weights = [1],
    num_simulation = 10,
    num_trading_days = 252*30)
    
    MC_thirtyyear.calc_cumulative_return()
    
    simulation_df, plot_title = MC_thirtyyear.plot_simulation()
    line_plot = simulation_df.hvplot(title=plot_title, legend=False).opts(yformatter="%.0f")
    
    """Fear & Greed Index results."""

    #hti = Html2Image()
    #url_data = 'https://money.cnn.com/data/fear-and-greed/'
    #hti.screenshot(url='https://money.cnn.com/data/fear-and-greed/', save_as='fear_reed.png')
    ## Importing Image class from PIL modul
    ## Opens a image in RGB mode
    #im = Image.open(r"fear_reed.png")
    ## Setting the points for cropped image
    #left = 470
    #top = 80
    #right = 1100
    #bottom = 400
    ## Cropped image of above dimension
    ## (It will not change original image)
    #im1 = im.crop((left, top, right, bottom))
    ## Shows the image in image viewer
    #im1.save("pic.png" , format="png")
    #fng_fig = pn.pane.PNG("pic.png",alt_text='F&G' , width = 600)

    #get the url to scrape from cnn
    url = 'https://money.cnn.com/data/fear-and-greed'
    res = rq.get(url)
    #create a soup object to parse the page
    soup = BeautifulSoup(res.content, 'html.parser')
    # create an etreee object to use Xapth attribute
    object_tree = etree.HTML(str(soup))

    # create and empty list to store the data being scraped
    list_fng = []

    # loop over the data and store in the list , this will grab the fear and greed tabulated data from the page
    for i in range(5):
        list_fng.append([object_tree.xpath('/html/body/div[3]/div[1]/div[1]/div[3]/div/div[1]')[0][0][i].text])

    df_fng = pd.DataFrame(list_fng)
    for i in df_fng:
        df_fng['Value'] = df_fng[i].str.extract(pat ='([0-9]{2})')
        df_fng['Status'] = df_fng[i].str.extract(pat= '((?<=\().+?(?=\)))')
        df_fng['Time'] = df_fng[i].str.replace(pat='(\:.*$)' , repl = '')

    df_fng.drop(columns=0 , inplace=True)

    fng_gauge = pn.indicators.Gauge(
        name='Fear and Greed Index',
        value=int(df_fng['Value'][0]),
        bounds=(0, 100), format='{value} %',
        colors=[(0.25, 'red'), (0.5, 'orange'), (0.75, 'yellow') , (1 , 'green')])


    """ Bot recomendation based on the metrics"""
    
    def bot():
        rsi = df_ticker['RSI']
        pe = df_metrics_long['value'][15]
        fear_greed = df_fng['Value'][0]
        """
        score = 0 
        
        if rsi > 70:
            score+=1
        elif rsi <30:
            score-=1
        else:
             score+=0           
        if pe > 30:
            score-=1
        elif pe>15:
            score+=1
        else:
            score+=0            
        if fear_greed > 75:
            score+=1
        elif fear_greed <30:
            score-=1
        else:
            score+=0 """
        #msg = f'The final score based on the {score} with the follwing paramaters used:\nRSI was {rsi}\nPERatio was {pe}\nFear and Greed Index was {fear_greed}'
        
        return "Still a work in progress"
            
        
    
    
    """ Layout Design of the dashboard"""
    # Layout design 
    twitter = pn.Row(pn.widgets.DataFrame(largest_sen_tweets,name='High Sentiment Tweets'),
                    pn.widgets.DataFrame(smallest_sen_tweets,  name='Low Sentiment Tweets'))

    fng_word = pn.Row(pie_pane , pn.pane.Matplotlib(fig) , fng_gauge)
    
    
    sentiment = pn.Column(fng_word ,twitter)
     
   
    Data_panel = pn.Tabs(
        ('Charts' , final_fig),
        ('Stock Metrics Data' , metric_table),
        ('Simulation' , line_plot),
        ('Sentiment' , sentiment),
        ('Recomendation' , bot()))
    
          
    return Data_panel

selection_panel = pn.Column(text_field,no_tweets,button)
Complete_dashboard = pn.Row(selection_panel,tabs)

(pn.template.VanillaTemplate(
    site="Group 1",
    title="Stock Analysis and Sentiment Dashboard" ,
    main=[""" 
    Group 1 Team Members:\n[Jihad Al-Hussain]\t[John Gaffney]\t[Shanel Kuchera]\t[Kazuki Takehashi]\t[Patrick Thornquist] """ , Complete_dashboard])).servable()