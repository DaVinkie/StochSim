import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_samples(N, xMin=-2, xMax=1, yMin=-1, yMax=1):
    ''' Returns an array of uniformly sampled N coordinates in a given range.
    '''
    x = np.random.uniform(xMin, xMax, N)
    y = np.random.uniform(yMin, yMax, N)
    samples = np.array([x, y])
    #Transpose, such that we can index on coordinate pairs easily
    return samples.T

def mandelbrot_area(i,j, max_iteration):
    ''' Determines if a given coordinate is in the mandelbrot set.
    It does this by checking if the corresponding irrational number tends to infinity in a given amount 
    of iterations.
    '''
    x = 0.0
    y = 0.0
    iteration = 0
    while(x*x + y*y <= 2*2 and iteration < max_iteration):
        xtemp = x*x - y*y + i
        y = 2*x*y + j
        x = xtemp 
        iteration+=1
    if (iteration == max_iteration):
        return True
    else:
        return False

def monte_carlo_area(sample_size, max_iterations, area):
    ''' Approximates the area of the mandelbrot set.
    The function generates samples and for each sample determines if it is in the mandelbrot set.
    The ratio of samples in the set can then be used to approximate its area.
    '''
    samples = make_samples(sample_size)
    ratio = []
    for i in samples:
        x = mandelbrot_area(i[0], i[1], max_iterations)
        ratio.append(x)

    # Uses a mask to determine the ratio of hits by Boolean indexing.
    ones = np.ones(S)
    ones = ones[ratio]
    total = np.sum(ones)
    # Proportion of True to False, multiplied by area of the sampled square.
    mandel_area = (total/S) * area
    return mandel_area


points=100
max_iteration = 30
colors = plt.cm.magma_r(np.linspace(0, 1, max_iteration+1))
fig=plt.figure(figsize=(20,15))

z = 0
xline = np.linspace(-2.5,1, int(points*1.75))
yline = np.linspace(-1,1, points)
index=0
for i in xline:
    for j in yline:
        x = 0.0
        y = 0.0
        iteration = 0
        while(x*x + y*y <= 2*2 and iteration < max_iteration):
            xtemp = x*x - y*y + i
            y = 2*x*y + j
            x = xtemp 
            iteration+=1
            
        #col = (iteration / max_iteration)        
        plt.scatter(i,j, color = colors[iteration], norm=mpl.colors.LogNorm(), s=100)   
plt.show()

TOTAL_AREA = 3*2
S = 10000
MAX_ITERATIONS = 100

ar = monte_carlo_area(S, MAX_ITERATIONS, TOTAL_AREA)
print("So estimated Area of the Mandelbrot Island is: ", ar)

