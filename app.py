import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, render_template
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error


app = Flask(__name__)



def AddZscore(df):
   df["ZScore"] = (df["Sales Amt"] - df["Sales Amt"].mean())/df["Sales Amt"].std()
   return df


#load data
data = pd.read_csv("./data/Data.csv")
data["PO Date"] = pd.to_datetime(data["PO Date"])
data["MonthStart"] = data["PO Date"].to_numpy().astype('datetime64[M]')
sales = data[["MonthStart","Sales Amt"]].groupby(["MonthStart"]).sum().reset_index()


deductionsData = data[data["Sales Amt"] < 0]
SalesData = data[data["Sales Amt"] >= 0]
deductionsData = AddZscore(deductionsData)
SalesData = AddZscore(SalesData)
deductionsDataNoOutliers = deductionsData[np.abs(deductionsData["ZScore"]) < 3]
SalesDataNoOutliers = SalesData[np.abs(SalesData["ZScore"]) < 3]

CombinedData = pd.concat([deductionsDataNoOutliers,SalesDataNoOutliers])
CombinedDatamonthly = CombinedData[["MonthStart","Sales Amt"]].groupby(["MonthStart"]).sum().reset_index()

#decomposition

decompose_result = seasonal_decompose(CombinedDatamonthly['Sales Amt'],model='multiplicative',period=6)

@app.route('/original.png')
def plot_png():
   fig = create_figure_original()
   print("fig here")
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')

@app.route('/NoOutliers.png')
def plot_boxOrig():
   fig = create_figure_NoOutliers()
   print("fig here")
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')

@app.route('/decompose.png')
def plot_decompose():
   fig = decompose()
   print("fig here")
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')

@app.route('/rolling.png')
def rolling():
   fig = rolling()
   print("fig here")
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')



def create_figure_original():
   fig = Figure(figsize=(10,5))
   
   axis = fig.add_subplot(1, 1, 1)

   xs = sales["MonthStart"].to_numpy()
   ys = sales["Sales Amt"].to_numpy()
   fig.suptitle('Original Monthly Sales', fontsize=16)

   axis.plot(xs, ys)

   return fig

def create_figure_NoOutliers():
   fig = Figure(figsize=(10,5))
   
   axis = fig.add_subplot(1, 1, 1)

   xs = CombinedDatamonthly["MonthStart"].to_numpy()
   ys = CombinedDatamonthly["Sales Amt"].to_numpy()
   fig.suptitle('Monthly Sales without Outliers', fontsize=16)
   axis.plot(xs, ys)

   return fig

@app.route('/HWES3.png')
def plot_HWES3():
   fig = HWSE3()
   print("fig here")
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')


@app.route('/forecast.png')
def plot_forecast():
   fig = forecast()
   print("fig here")
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')

def decompose():  
   fig, axs = plt.subplots(3,figsize=(10,5))
   fig.suptitle('Decomposition')
      
   axs[0].plot(CombinedDatamonthly['MonthStart'],decompose_result.trend)
   axs[1].plot(CombinedDatamonthly['MonthStart'],decompose_result.seasonal)
   axs[2].scatter(CombinedDatamonthly['MonthStart'],decompose_result.resid)

   labels = ["Trend","Seasonal","Residuals"]
   i = 0
   for ax in axs.flat:
       ax.set(ylabel=labels[i])
       i+=1

   return fig

def rolling():
   
   residuals = decompose_result.resid
   rolmean = residuals.rolling(6).mean()
   rolstd = residuals.rolling(6).std()
   
   fig = Figure(figsize=(10,5))
   
   ax = fig.add_subplot(111)

   fig.suptitle('Rolling Mean and Standard Deviation')
      
   ax.plot(residuals,label = "Residuals")
   ax.plot(rolmean,label = "Rolling Mean")
   ax.plot(rolstd,label = "Rolling STD")
   ax.legend()

   return fig


def HWSE3():
   fig = Figure(figsize=(10,5))
   
   ax = fig.add_subplot(111)

   CombinedDatamonthly["HWES3_ADD"] = ExponentialSmoothing(CombinedDatamonthly["Sales Amt"],trend="add",seasonal="add",seasonal_periods=12).fit().fittedvalues
   CombinedDatamonthly["HWES3_MUL"] = ExponentialSmoothing(CombinedDatamonthly["Sales Amt"],trend="mul",seasonal="mul",seasonal_periods=12).fit().fittedvalues
   
   fig.suptitle('HWES3')
    

   ax.plot(CombinedDatamonthly['MonthStart'],CombinedDatamonthly['Sales Amt'],label = "Sales")
   ax.plot(CombinedDatamonthly['MonthStart'],CombinedDatamonthly['HWES3_ADD'],label = "HWES3_ADD")
   ax.plot(CombinedDatamonthly['MonthStart'],CombinedDatamonthly['HWES3_MUL'],label = "HWES3_MUL")
   ax.legend()

   return fig


def forecast():
   # Split into train and test set
   train = CombinedDatamonthly[:30]
   test = CombinedDatamonthly[30:]

   #
   fitted_model = ExponentialSmoothing(train["Sales Amt"],trend='add',seasonal='add',seasonal_periods=12).fit()

   #for metrics to be used later
   test_for_metrics = fitted_model.forecast(6)
   test_predictions = fitted_model.forecast(30) #24 months ahead

   CombinedDatamonthly['Sales Amt'].plot(legend=True,label='TRAIN')
   #test['Sales Amt'].plot(legend=True,label='TEST',figsize=(6,4))
   test_predictions.plot(legend=True,label='PREDICTION')


   mae = mean_absolute_error(test["Sales Amt"],test_for_metrics)
   mse = mean_squared_error(test["Sales Amt"],test_for_metrics)

   fig = Figure(figsize=(10,5))
   
   ax = fig.add_subplot(111)
   fig.suptitle('Holt Winters Exponential Smoothing Forecast')

   d = pd.date_range(start='2021-07-01', periods=30, freq='MS')    

   ax.plot(CombinedDatamonthly['MonthStart'],CombinedDatamonthly['Sales Amt'],label = "Sales")
   ax.plot(d,test_predictions,label = "Forecast")
   ax.legend()

   ax.set_xlabel('MAE = ' + str(mae) + '\n MSE = ' + str(mse))
   return fig

@app.route('/plot')
def plot():
   return render_template('plot.html', name = plt.show())

@app.route('/')
def index():
   return render_template('plot.html')

   

if __name__ == '__main__':
   app.run(debug = True)