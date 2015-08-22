import pandas.io.data as dataReader
import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plt
import datetime as dt
import sys as sys


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


def gradAscent(rho,maPos,x,deltaDSR,R):
	temp = rho*deltaDSR*maPos*R
        #print(rho,x,deltaDSR,R)
	return [temp*i for i in x]


"""def model(n,F,r,t):

    w = gredientAscent()
    F[t] = np.tanh(np.dot(w,x))
    return F[t]"""

if __name__ == '__main__':

    #getting the stock data from yahoo
    apple = closingPriceData('AAPL');

    #different parameters for the simulation
    #max position are the number of maximum stock agent can buy
    #features is window size of past returns that are accounted for decision of next position
    #rho is learning rate of gradinet ascent algo
    maxPosition = 20
    nFeatures = 10
    rho = 0.5
    totalData = len(apple)
    #calculating return per stock
    r = map(lambda a,b:a-b,apple[1:],apple[:-1])
    r.insert(0,0)
    #print r
    #R is net return which is product of maxPoistion*r*F[t-1]
    #first return will be zero, since first position gives output as second return
    #returns array is offset by 1 as compared to positions array
    R=[0]
    R =np.array(R,dtype=float)
    #first position i am taking is full buy
    F = np.array([1],dtype=float)

    #x is the feature vector
    x = []
    #w is the parameter vector which will be tuned to maximize the profit
    w = np.zeros(2,dtype=float)
    #temperary variable for ith position
    F_t=np.array([],dtype=float)

    #this loop we are running before we have proper window size
    #these will be exactly nFeature+2 elements in the feature vector, nFeature number of returns and two other elements
    #we wont have this size of window since the start itself
    #so we have to slowly increase the window size
    for t in range(1,nFeatures+1):
        R = np.append(R,[r[t]*F[t-1]*maxPosition])
        #calculating exponential weighted mean average for first and second moment of Rt
        secondMoment = np.array( [i*i for i in R],dtype=float)
        firstMomentEWMA = pd.ewma(R,span=t)
        secondMomentEWMA = pd.ewma(secondMoment,span=t)

        #calculate gradient of differential sharp Ratio
        deltaDSR = secondMomentEWMA[-2]-firstMomentEWMA[-2]*R[-1]/ math.pow(secondMomentEWMA[-2]-math.pow(firstMomentEWMA[-2],2),1.5)
        #x here is feature vector
        #contains last n returns per stock, and t-1th position
        x=np.append([1,F[-1]],np.array(R[-int(t):],dtype=float))
        w = np.append(w,np.array(1,dtype=float))
        #call Gradient asecent with repect to w here
        w1 = np.array(np.zeros(w.size),dtype=float)
        if not t==1:
            for num in range(100):
                #w1 = w+gradAscent(rho,maxPosition,x,deltaDSR,R[-1])
                w1 = w+ gradAscent(rho,maxPosition,x,deltaDSR,R[-1])
                """out = np.dot(w1-w,w1-w)
                w = w1
                if out < 0.01:
                    break"""
        else:
            pass
        #print w,x
        F_t = np.tanh(np.dot(w,x))
        F = np.append(F,F_t)

    #sys.exit()
    t = nFeatures+1
    w = np.array(w,dtype=float)
    while(t<totalData):
        R = np.append(R,[r[t]*F[t-1]*maxPosition])
        #calculating exponential weighted mean average for first and second moment of Rt
        secondMoment = [i*i for i in R]
        secondMoment = np.array(secondMoment,dtype=float)
        firstMomentEWMA = pd.ewma(R,span=nFeatures)
        secondMomentEWMA = pd.ewma(secondMoment,span=nFeatures)

        #x here is feature vector
        #contains last n returns per stock, and t-1th position
        x=np.append([1,F[-1]],np.array(R[-int(nFeatures):],dtype=float))
        #no need to increase feature vector here since we have reached proper window size
        #w = np.append(w,np.zeros(1,dtype=float))
        #calculate gradient of differential sharp Ratio
        deltaDSR = secondMomentEWMA[-2]-firstMomentEWMA[-2]*R[-1]/ math.pow(secondMomentEWMA[-2]-math.pow(firstMomentEWMA[-2],2),1.5)

        #call Gradient asecent with repect to w here
        w1 = np.array(np.zeros(nFeatures+2),dtype=float)
        for num in range(100):

            w1 = w+gradAscent(rho,maxPosition,x,deltaDSR,R[-1])
            out = np.dot(w1-w,w1-w)
            w = w1
            if out < 0.01:
                break

        F_t = np.tanh(np.dot(w,x))
        F = np.append(F,F_t)
        t = t+1

    cumR =0
    for i in range(R.size):
        cumR = sum(R[:i])
        print cumR
    #print len(F),R
    #actF = [a*20 for a in F]
    #plt.plot(cumR,'r')
    #plt.plot(F,'g')
    #plt.show()
    #plt.plot(R,'g')
    #plt.plot(cumR,'r')
    plt.show()
    print(R)






