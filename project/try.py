import pandas.io.data as dataReader
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import datetime as dt

def getData(stock):
    """ Returns apple data between start and end data"""
    start = dt.datetime(2012,1,1)
    end = dt.datetime(2014,1,1)
    apple = dataReader.DataReader(stock,'yahoo',start,end)
    return apple

def closingPriceData(stock):
    """return list of closing prices"""
    stock = getData(stock)
    sizeList = stock['Adj Close'].size
    closePriceList =[]
    for i in range(0,sizeList):
        closePriceList.append(stock['Adj Close'][i])
    return closePriceList

def plotG(priceList):
    plt.plot(priceList)
    plt.show()

if __name__ == '__main__':
	price = getData('AAPL')
	priceClose = price['Adj Close']
	squaredPrice = [i*i for i in priceClose]
	squaredPrice = np.array(squaredPrice,dtype=float)
	plt.plot(priceClose)
	#plotG(price)
	rs = pd.ewma(priceClose,span=20)
	ewmaPrice2 = pd.ewma(squaredPrice,span=20)
	plt.plot(priceClose)
	plt.plot(rs)
	plt.plot(ewmaPrice2)
	plt.show()