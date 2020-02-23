import utilities as util
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import random

# Calculate the Least Squares Regression.
def leastSquares(xs, ys, lineType):
    ones = np.ones(xs.shape)
    if lineType == "linear":
        X = np.column_stack((ones, xs))
    elif lineType == "quad":
        X = np.column_stack((ones, xs, xs**2))
    elif lineType == "cubic":
        X = np.column_stack((ones, xs, xs**2, xs**3))
    A = np.linalg.inv(X.T @ X) @ X.T @ ys
    return A

# Calculate the residual error with y-squared differences.
def ySquared(xs, ys, A, lineType):
    if lineType == "linear":
        ydiffs = np.poly1d([A[1],A[0]])(xs) - ys
    elif lineType == "quad":
        ydiffs = np.poly1d([A[2],A[1],A[0]])(xs) - ys
    elif lineType == "cubic":
        ydiffs = np.poly1d([A[3],A[2],A[1],A[0]])(xs) - ys

    squared = ydiffs**2
    return np.sum(squared)

# Calculate the residual error.
def resError(xs, ys, AL, AQ, AC):
     return [ySquared(xs, ys, AL, "linear"), ySquared(xs, ys, AQ, "quad"), ySquared(xs, ys, AC, "cubic")]

# Load the points from the file specified in the command line.
points = util.load_points_from_file(sys.argv[1])
xpoints = points[0]
ypoints = points[1]

# Calculate the number of 20-point chunks.
noOfChunks = len(xpoints) // 20

# Split the points into chunks.
xs = []
ys = []
xsshuffled = []
ysshuffled = [] 
for i in range(noOfChunks):
    xs.append(xpoints[i*20 : (i+1)*20])
    ys.append(ypoints[i*20 : (i+1)*20])
    # Shuffle the data.
    shuffle = np.arange(xs[i].shape[0])
    np.random.shuffle(shuffle)
    xsshuffled.append(xs[i][shuffle])
    ysshuffled.append(ys[i][shuffle])

# Separate the training and test data.
xstest = []
xstrain = []
ystest = []
ystrain = []
for i in range(noOfChunks):
    xstest.append(xsshuffled[i][:5])
    xstrain.append(xsshuffled[i][15:])
    ystest.append(ysshuffled[i][:5])
    ystrain.append(ysshuffled[i][15:])

# Assign the least squares results to a matrix A.
ALinear = []
AQuad = []
ACubic = []
for i in range(noOfChunks):
    ALinear.append(leastSquares(xstrain[i], ystrain[i], "linear"))
    AQuad.append(leastSquares(xstrain[i], ystrain[i], "quad"))
    ACubic.append(leastSquares(xstrain[i], ystrain[i], "cubic"))

# Assign the residual errors into an array.
resErrors = []
for i in range(noOfChunks):
    resErrors.append(resError(xstest[i], ystest[i], ALinear[i], AQuad[i], ACubic[i]))

# Plot the line of best fit.
# The colour represents the line type:
#   - linear        = blue
#   - quadratic     = green
#   - cubic         = yellow
def plotBestFit(xs, ys, AL, AQ, AC, resErrors):
    if min(resErrors) == resErrors[0]:
        plt.plot(xs, np.poly1d([AL[1],AL[0]])(xs), color="b")
    elif min(resErrors) == resErrors[1]:
        plt.plot(xs, np.poly1d([AQ[2],AQ[1],AQ[0]])(xs), color="g")
    elif min(resErrors) == resErrors[2]:
        plt.plot(xs, np.poly1d([AC[3],AC[2],AC[1],AC[0]])(xs), color="y")

# Plot the graph.
for i in range(noOfChunks):
    plotBestFit(xs[i], ys[i], ALinear[i], AQuad[i], ACubic[i], resErrors[i])
util.view_data_segments(xpoints, ypoints)