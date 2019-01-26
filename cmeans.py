import sys
import numpy as np
import random

if len(sys.argv) != 5:
    print('usage: ', sys.argv[0], 'data_file k r output_file')
    sys.exit()

filename=open(sys.argv[1],"r")
data=np.loadtxt(filename,delimiter=',')
k = int(sys.argv[2])
r = int(sys.argv[3])
outputFile = sys.argv[4]
p = 2
epsilon = 10**-5
minQuant = 10**-10
n,m = data.shape
if(k>n):
    print("k is greater than number of points. Exiting...")
    sys.exit()

def equalCenters(centers1,centers2):
    for i in range(k):
        dist=np.linalg.norm(centers1[i]-centers2[i])
        if(dist!=0):
            return False
            break
    return True

def calculateCij():
    calculateDij()
    calculateFi()
    calculateGij()
    for i in range(n):
        for j in range(k):
            cij[i][j]= 1/gij[i][j]

def calculateDij():
    for i,point in enumerate(data):
        for j in range(k):
            dij[i][j] = np.linalg.norm(point-centers[j])**2

def calculateFi():
    for i in range(n):
        fi[i] = 0
        for t in range(k):
            fi[i] = fi[i] + (1/((dij[i][t])**(1/(p-1))))

def calculateGij():
    for i in range(n):
        for j in range(k):
            gij[i][j] = fi[i] * (dij[i][j]**(1/(p-1)))

def calculateCenters():
    for j in range(k):
        numerator = 0
        denominator = 0
        for i in range(n):
            numerator = numerator+(cij[i][j]**p)*data[i]
            denominator = denominator + (cij[i][j]**p)

        centers[j] = numerator/denominator

def tolerance(cold,cnew):
	error = np.linalg.norm(cold-cnew)
	if(error < epsilon):
		return True
	else:
		return False

def checkMinQuantError(quantError):
	if(quantError < epsilon):
		return True
	else:
		return False

minQuantError = sys.maxsize

for itr in range(r):
    print("\nIteration: ",itr+1,'\n')
    centers = np.zeros((k,m))
    previousCenters =  np.zeros((k,m))
    cij = np.zeros((n,k))
    dij = np.zeros((n,k))
    fi = np.zeros((n,1))
    gij = np.zeros((n,k))
    clusterId = np.zeros((n,1))
    rand = random.sample(range(0,n),k)
    cijold = np.zeros((n,k))
    cijnew = np.zeros((n,k))
    
    for i in range(n):
    	rand = np.random.dirichlet(np.ones(k),size=1)
    	for j in range(k):
    		cij[i][j] = rand[0][j]

    calculateCenters()

    isOptimal1 = False 
    isOptimal2 = False
    isOptimal3 = False

    while (not isOptimal1 and not isOptimal2 and not isOptimal3):
        
        previousCenters = np.copy(centers)
        cijold = np.copy(cij)
        calculateCij()
        cijnew = np.copy(cij)
        calculateCenters()
        calculateDij()
        quantizationError = 0
        for i in range(n):
            for j in range(k):
                quantizationError = quantizationError + (cij[i][j]**p)*dij[i][j]

        print("QuantizationError: ",quantizationError)
        isOptimal1 = equalCenters(centers,previousCenters)
        isOptimal2 = tolerance(cijold,cijnew)
        isOptimal3 = checkMinQuantError(quantizationError)
 
    for i in range(n):
        max = float('-inf')
        for j in range(k):
            if(cij[i][j]>max):
                max = cij[i][j]
                clusterId[i] = j

    if(quantizationError < minQuantError):
        minQuantError = quantizationError
        finalClusters = np.copy(clusterId)

np.savetxt(outputFile,finalClusters,delimiter=',')
print("\nBest quantization error obtained in ",r," iterations: ",minQuantError)