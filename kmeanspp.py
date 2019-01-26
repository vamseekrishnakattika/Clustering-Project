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

n,m = data.shape
if(k>n):
    print("k is greater than number of points. Exiting...")
    sys.exit()

def minDistance(point):
	minDist = sys.maxsize
	for j,center in enumerate(centers):
		dist = np.linalg.norm(center-point)
		if(dist < minDist):
			minDist = dist
	return minDist

def equalCenters(centers1,centers2):
    for i in range(k):
        dist=np.linalg.norm(centers1[i]-centers2[i])
        if(dist!=0):
            return False
            break
    return True

minQuantError = sys.maxsize

for itr in range(r):
    print("\nIteration: ",itr+1,'\n')   
    centers = np.zeros((k,m))
    previousCenters =  np.zeros((k,m))    
    clusterId = np.zeros((n,1))

    rand = random.randint(0,n-1)
    centers[0] = data[rand]
    for j in range (k-1):
        maxDist = float('-inf')
        for i,point in enumerate(data):
            dist = minDistance(point)
            if(dist > maxDist):
                maxDist = dist
                centers[j+1] = data[i]
                   
    isOptimal = False 

    while (not isOptimal):
        for i, point in enumerate(data):
            minDist = sys.maxsize
            for j, center in enumerate(centers):
                dist = np.linalg.norm(center-point)
                if(dist < minDist):
                    minDist = dist
                    clusterId[i] = j

        previousCenters = np.copy(centers)
        centers = np.zeros((k,m))
        
        for idx in range(k):
            count=0
            for i,xt in enumerate(clusterId):
                if(xt == idx):
                    centers[idx] += data[i]
                    count += 1
            if(count > 0):
                centers[idx] = centers[idx] / count
       
        quantizationError = 0
        for idx in range(k):
            for i,xt in enumerate(clusterId):
                if(xt == idx):
                    quantizationError += np.linalg.norm(centers[idx]-data[i])**2

        print("QuantizationError: ",quantizationError)
        isOptimal = equalCenters(centers,previousCenters)
    
    if(minQuantError > quantizationError):
           minQuantError = quantizationError
           finalClusters = np.copy(clusterId)
np.savetxt(outputFile, finalClusters, delimiter=',')
print("\nBest quantization error obtained in ",r," iterations: ",minQuantError)