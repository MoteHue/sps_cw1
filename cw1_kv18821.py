import utilities as util
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# Calculate the Least Squares Regression.
def leastSquaresLinear(xs, ys):
    ones = np.ones(xs.shape)
    X = np.column_stack((ones, xs))
    A = np.linalg.inv(X.T @ X) @ X.T @ ys
    return A
def leastSquaresQuad(xs, ys):
    ones = np.ones(xs.shape)
    X = np.column_stack((ones, xs, xs**2))
    A = np.linalg.inv(X.T @ X) @ X.T @ ys
    return A
def leastSquaresCubic(xs, ys):
    ones = np.ones(xs.shape)
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
    #elif lineType == "exp":
    #    ydiffs = (A[0] + A[4]*np.exp(xs)) - ys

    squared = ydiffs**2
    return np.sum(squared)

# Calculate the residual error.
def resError(xs, ys, AL, AQ, AC):
     return [ySquared(xs, ys, AL, "linear"), ySquared(xs, ys, AQ, "quad"), ySquared(xs, ys, AC, "cubic")]#, ySquared(xs, ys, A, "exp")]

# Load the points from the file specified in the command line.
points = util.load_points_from_file(sys.argv[1])
xpoints = points[0]
ypoints = points[1]

# Calculate the number of 20-point chunks.
noOfChunks = len(xpoints) // 20

# Split the points into chunks.
xs = []
ys = []
for i in range(noOfChunks):
    xs.append(xpoints[i*20 : (i+1)*20])
    ys.append(ypoints[i*20 : (i+1)*20])

# Assign the least squares results to a matrix A.
ALinear = []
for i in range(noOfChunks):
    ALinear.append(leastSquaresLinear(xs[i], ys[i]))
AQuad = []
for i in range(noOfChunks):
    AQuad.append(leastSquaresQuad(xs[i], ys[i]))
ACubic = []
for i in range(noOfChunks):
    ACubic.append(leastSquaresCubic(xs[i], ys[i]))

# Assign the residual errors into an array.
resErrors = []
for i in range(noOfChunks):
    resErrors.append(resError(xs[i], ys[i], ALinear[i], AQuad[i], ACubic[i]))

# Plot the line of best fit.
# The colour represents the line type:
#   - linear        = blue
#   - quadratic     = green
#   - cubic         = yellow
#   - exponential   = red
def plotBestFit(xs, ys, AL, AQ, AC, resErrors):
    if min(resErrors) == resErrors[0]:
        plt.plot(xs, np.poly1d([AL[1],AL[0]])(xs), color="b")
    elif min(resErrors) == resErrors[1]:
        plt.plot(xs, np.poly1d([AQ[2],AQ[1],AQ[0]])(xs), color="g")
    elif min(resErrors) == resErrors[2]:
        plt.plot(xs, np.poly1d([AC[3],AC[2],AC[1],AC[0]])(xs), color="y")
    #elif min(resErrors) == resErrors[4]:
    #    plt.plot(xs, A[0] + A[4]*np.exp(xs), color="r")

# Plot the graph.
for i in range(noOfChunks):
    plotBestFit(xs[i], ys[i], ALinear[i], AQuad[i], ACubic[i], resErrors[i])
util.view_data_segments(xpoints, ypoints)