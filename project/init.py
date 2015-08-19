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


"""def model(n,F,r,t):

    w = gredientAscent()
    F[t] = np.tanh(np.dot(w,x))
    return F[t]"""

if __name__ == '__main__':

    #largest position that can be held
    maxPosition = 20
    nFeatures = 10

    apple = closingPriceData('AAPL');

    #calculating return per stock
    r = map(lambda a,b:a-b,apple[1:],apple[:-1])
    r.insert(0,0)

    R=[0]
    R =np.array(R,dtype=float)
    #first position i am taking is full buy
    F = [1]
    x = []
    for i in range(1,nFeatures+1):
        np.append(R,[r[i]*F[i-1]*maxPosition])
        #calculating exponential weighted mean average for first and second moment of Rt
        secondMoment = [i*i for i in R]
        secondMoment = np.array(secondMoment,dtype=float)
        firstMomentEWMA = pd.ewma(R,span=20)
        secondMomentEWMA = pd.ewma(secondMoment,span=20)

        #x here is feature vector
        #contains last n returns per stock, and t-1th position
        x=[1,R[-int(i):],F[int(i)-1]]

        #call Gradient asecent with repect to w here
        #gradAscent(rho,w,x,firstMomentEWMA,secondMomentEWMA,R)
        #update w =[] here
        #predict next F[i] = w.*x
        #continue
    t = nFeatures+1
    while(t<totalData):

        #do the above
        #perform gradient
        #update parameter
        #get next position

    #for last sell it all and calculate profit






