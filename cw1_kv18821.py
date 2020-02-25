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
    elif lineType == "sine":
        X = np.column_stack((ones, np.sin(xs)))
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
    elif lineType == "sine":
        ydiffs = np.poly1d([A[1],A[0]])(np.sin(xs)) - ys

    squared = ydiffs**2
    return np.sum(squared)

# Calculate the residual error.
def resError(xs, ys, AL, AQ, AC, AS):
     return [ySquared(xs, ys, AL, "linear"), ySquared(xs, ys, AQ, "quad"), ySquared(xs, ys, AC, "cubic"), ySquared(xs, ys, AS, "sine")]

# Load the points from the file specified in the command line.
points = util.load_points_from_file(sys.argv[1])
xpoints = points[0]
ypoints = points[1]

# Calculate the number of 20-point chunks.
noOfChunks = len(xpoints) // 20

# Split the points into chunks.
xs = []
ys = []
for n in range(noOfChunks):
    xs.append(xpoints[n*20 : (n+1)*20])
    ys.append(ypoints[n*20 : (n+1)*20])

xstrain = []
ystrain = []

for n in range(noOfChunks):
    xtrain = []
    ytrain = []
    for i in range(20):
        # Take out test point from the rest of the training data.
        xtr = []
        ytr = []

        xtr.extend(xs[n][0:i])
        xtr.extend(xs[n][i+1:20])
        ytr.extend(ys[n][0:i])
        ytr.extend(ys[n][i+1:20])

        xtrain.append(xtr)
        ytrain.append(ytr)
    xstrain.append(xtrain)
    ystrain.append(ytrain)

xstrain = np.array(xstrain)
ystrain = np.array(ystrain)

minResErrors = []
for n in range(noOfChunks):
    sum = 0
    for i in range(20):
        AL = leastSquares(xstrain[n][i], ystrain[n][i], "linear")
        AQ = leastSquares(xstrain[n][i], ystrain[n][i], "quad")
        AC = leastSquares(xstrain[n][i], ystrain[n][i], "cubic")
        AS = leastSquares(xstrain[n][i], ystrain[n][i], "sine")
        sum += np.array(resError(xs[n][i], ys[n][i], AL, AQ, AC, AS))
    if min(sum) == sum[0]:
        minResErrors.append([min(sum), "linear"])
    if min(sum) == sum[1]:
        minResErrors.append([min(sum), "quad"])
    if min(sum) == sum[2]:
        minResErrors.append([min(sum), "cubic"])
    if min(sum) == sum[3]:
        minResErrors.append([min(sum), "sine"])

# Plot the line of best fit.
# The colour represents the line type:
#   - linear        = blue
#   - quadratic     = green
#   - cubic         = yellow
#   - sine          = red
def plotBestFit(xs, ys, resError):
    if resError[1] == "linear":
        A = leastSquares(xs, ys, "linear")
        plt.plot(xs, np.poly1d([A[1],A[0]])(xs), color="b")
    elif resError[1] == "quad":
        A = leastSquares(xs, ys, "quad")
        plt.plot(xs, np.poly1d([A[2],A[1],A[0]])(xs), color="g")
    elif resError[1] == "cubic":
        A = leastSquares(xs, ys, "cubic")
        plt.plot(xs, np.poly1d([A[3],A[2],A[1],A[0]])(xs), color="y")
    elif resError[1] == "sine":
        A = leastSquares(xs, ys, "sine")
        plt.plot(xs, np.poly1d([A[1],A[0]])(np.sin(xs)), color="r")

sumErrors = 0
for n in range(noOfChunks):
    sumErrors += minResErrors[n][0]

print(sumErrors)

# Plot the graph.
if (len(sys.argv) > 2) :
    if (sys.argv[2] == "--plot") :
        for n in range(noOfChunks):
            plotBestFit(xs[n], ys[n], minResErrors[n])
        util.view_data_segments(xpoints, ypoints)