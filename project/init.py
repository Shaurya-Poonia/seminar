import pandas.io.data as dataReader
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

def model(n,F,r,t):
    appleClose = closingPriceData('AAPL')

    #modelling and parameter adjustment will beb done here
    x[:]=[1,r[-n:],F[t-1]]
    w = gredientAscent()
    F[t] = np.dot(w,x,)
    return F[t]

if __name__ == '__main__':
    apple = closingPriceData('AAPL');

    #calculating returns
    list1 = range(1,10)
    r = map(lambda a,b:a-b,apple[1:],apple[:-1])
    F =[]
    for i in range(500):
        F.append(model(n,F,r,i));

    plotG(F)
    plotG(r)




