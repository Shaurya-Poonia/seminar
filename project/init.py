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



def gradAscent(rho,maxPosition,friction,currentPos,deltaDSR,xt,x,wt,rt):
    sign = (currentPos-xt[1])/abs(currentPos-xt[1])
    
    temp = sign*(1-currentPos*currentPos)
    tempList1 = np.array([-1*friction*temp for a in xt],dtype=float)
    temp = -1*temp*wt[1]*friction+rt+friction*sign
    tempList2 = np.array([a*temp for a in x],dtype=float)
    #print len(tempList1),len(tempList2)
    tempList = np.array(tempList1+tempList2,dtype=float)
    tempList = [maxPosition*deltaDSR*a*rho for a in tempList]
    if(math.isnan(np.tanh(np.dot(wt,tempList)))):
        sys.exit()
    print("******************************************")
    print deltaDSR
    print tempList

    print("******************************************")
    return np.array(tempList,dtype=float) 


"""def gradAscent(rho,maPos,x,deltaDSR,R):
	temp = rho*deltaDSR*maPos*R
        #print(rho,x,deltaDSR,R)
	return [temp*i for i in x]"""


if __name__ == '__main__':

    #getting the stock data from yahoo
    apple = closingPriceData('^GSPC');

    #different parameters for the simulation
    #max position are the number of maximum stock agent can buy
    #features is window size of past returns that are accounted for decision of next position
    #rho is learning rate of gradinet ascent algo
    maxPosition = 1
    nFeatures = 10
    rho = 0.5
    friction = 0.01
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
    F = np.array([1,1],dtype=float)

    #x1 is the feature vector at time t-1
    x = []
    #x is the feature vector at time t
    xt = []
    
    #wt is the parameter vector which will be tuned to maximize the profit at time t
    wt = np.array([1,1],dtype=float)
    #temperary variable for ith position
    F_t=np.array([],dtype=float)

    #this loop we are running before we have proper window size
    #these will be exactly nFeature+2 elements in the feature vector, nFeature number of returns and two other elements
    #we wont have this size of window since the start itself
    #so we have to slowly increase the window size
    R = np.append(R,r[1]*F[0]*maxPosition)
    for t in range(2,nFeatures):
        R = np.append(R,[r[t]*F[-1]*maxPosition])
        #calculating exponential weighted mean average for first and second moment of Rt
        secondMoment = np.array( [i*i for i in R],dtype=float)
        firstMomentEWMA = pd.ewma(R,span=t)
        secondMomentEWMA = pd.ewma(secondMoment,span=t)

        #calculate gradient of differential sharp Ratio
        deltaDSR = secondMomentEWMA[-2]-firstMomentEWMA[-2]*R[-1]/ math.pow(secondMomentEWMA[-2]-math.pow(firstMomentEWMA[-2],2),1.5)
        print firstMomentEWMA
        print secondMomentEWMA
        print deltaDSR
        #x here is feature vector
        #contains last n returns per stock, and t-1th position
        x=np.append([1,F[-2]],np.array(R[-int(t):-1],dtype=float))
        xt = np.append([1,F[-1]],np.array(R[-int(t-1):],dtype=float))
        wt = np.append(wt,np.array(1,dtype=float))
        #call Gradient asecent with repect to w here
        w1 = np.array(np.zeros(wt.size),dtype=float)
        #print wt.size,x.size,xt.size
    
        F_t = 0.5
        for num in range(100):
            print wt[1]
            w1= wt+ gradAscent(rho,maxPosition,friction,F_t,deltaDSR,xt,x,wt,R[-1]/F[-1])
            wt = w1
            F_t = np.tanh(np.dot(wt,xt))
            print wt[1]
            sys.exit()
        F = np.append(F,F_t)

    t = nFeatures+2
    wt = np.array(wt,dtype=float)
    while(t<totalData):
        R = np.append(R,[r[t]*F[-1]*maxPosition])
        #calculating exponential weighted mean average for first and second moment of Rt
        secondMoment = [i*i for i in R]
        secondMoment = np.array(secondMoment,dtype=float)
        firstMomentEWMA = pd.ewma(R,span=nFeatures)
        secondMomentEWMA = pd.ewma(secondMoment,span=nFeatures)

        #x here is feature vector
        #contains last n returns per stock, and t-1th position
        x=np.append([1,F[-2]],np.array(R[-int(nFeatures+1):-1],dtype=float))
        xt=np.append([1,F[-1]],np.array(R[-int(nFeatures):],dtype=float))
        
        #no need to increase feature vector here since we have reached proper window size
        #w = np.append(w,np.zeros(1,dtype=float))
        #calculate gradient of differential sharp Ratio
        deltaDSR = secondMomentEWMA[-2]-firstMomentEWMA[-2]*R[-1]/ math.pow(secondMomentEWMA[-2]-math.pow(firstMomentEWMA[-2],2),1.5)
        print x,xt
        #call Gradient asecent with repect to w here
        w1 = np.array(np.zeros(nFeatures+2),dtype=float)
        F_t=0.5
        for num in range(100):
            w1= wt+ gradAscent(rho,maxPosition,friction,F_t,deltaDSR,xt,x,wt,R[-1]/F[-1])
            wt = w1
            F_t = np.tanh(np.dot(wt,xt))
            print w1,F_t,t
            if math.isnan(F_t):
                sys.exit()
            
        F = np.append(F,F_t)
        t = t+1

    cumR =[0]
    for i in range(R.size):
        cumR.append(sum(R[:i]))

    plt.plot(cumR,'r')
    plt.plot(R,'g')
    plt.plot(apple,'b')
    plt.plot([a-20 for a in F],'orange')
    print sum(R)
    plt.show()







