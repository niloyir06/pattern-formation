# Various parameters can be changed to change the pattern formed
# A new window will be opened to visualize the simulation
# Please press 'q' to stop simulation. 

import numpy as np
import math
import tqdm
from matplotlib import pyplot as plt
import imageio
import cv2


def calc_b(a,c,d):
    b = a*np.exp(np.array(d)**2/c)
    return b

def f_attraction(a):
    return a

def f_repulsion(x1, x2, a, b, c):
    
    fr = b * np.exp(-(np.linalg.norm(np.array(x1) - np.array(x2))**2)/c)
    return fr


def calc_f(x1, x2, a, b, c):
    f = -(np.array(x1) - np.array(x2)) * (f_attraction(a) - f_repulsion(x1, x2, a, b, c))
    return f

def update_v(v, f, w):
    new_v = w*np.array(v) + np.array(f)
    return new_v

def update_x(x, v, f, w):
    new_x = np.array(x) + update_v(v, f, w)
    return new_x



plt.rcParams['figure.dpi'] = 300
plt.ioff()
np.random.seed(0)           #Change random seed for different initializations
plotting_frequency = 25     #Frequency of plotting, every (plotting_frequency)th
                            #step is plotted. For speed purpose.

num_agents = 6              #Number of agents
iterations = 10000          #Max number of iterations to run
xmin = 0                    #Lower range for random initialization
xmax = 20                   #Upper range for random initialization


### Specifying different parameters
a = 0.0001
c = 0.1
d=[2]
b=calc_b(a,c,d)

w = 0
v = [0,0]                   #Initial value of v set to 0


###Randomly initialize location of different swarm agents
agents = {}
for i in range(num_agents):
    agents[i] = np.random.randint(xmin, xmax, 2)

###Initialize an empty dict that will contain f values between different agents
f = {}    
for i in agents:
    f[i] = {}


print("Initial position:", agents)
fig = plt.figure()
for i in agents:   
    plt.scatter(agents[i][0], agents[i][1], marker = '.')


### Perform simulation and save animation as "simulation.mp4"

with imageio.get_writer('simulation.mp4', mode='I', fps = 24) as writer:
    cv2.namedWindow("Simulation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Simulation", 800, 600)
    for n in tqdm.trange(iterations):
        for i in agents:
            for j in agents:
                if j > i and j != i:
                    f[i][j] = calc_f(agents[i], agents[j], a, b, c)
                if i>j:
                    f[i][j] = -f[j][i]
        
        fsum = {}
        for i in agents:
            fsum[i] = 0
            for j in agents:
                if j != i:
                    fsum[i]+=f[i][j]
        
        
        new_positions = {}
        for i in agents:
            new_positions[i] = update_x(agents[i], v, fsum[i], w)
        
        agents = new_positions
        
        if n%plotting_frequency == 0:         #Every xth step is plotted for speed purpose
            fig = plt.figure()
            plt.xlim([xmin,xmax])
            plt.ylim([xmin,xmax])
            for i in agents:   
                plt.scatter(agents[i][0], agents[i][1], marker = '.')
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(img)
            plt.close()
            
            cv2.imshow('Simulation', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    
    
    
    fig = plt.figure()
    plt.xlim([xmin,xmax])
    plt.ylim([xmin,xmax])
    for i in agents:     
        plt.scatter(agents[i][0], agents[i][1])
    
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    writer.append_data(img)
    cv2.destroyAllWindows() 

print("Final position:", agents)







