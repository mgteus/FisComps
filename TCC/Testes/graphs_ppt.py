import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import datetime
import time
import collections
import random
import math





def sigmoid(x):
  return 1 / (1 + math.exp(-x))

fig, ax = plt.subplots()
#plt.scatter(x_rand, y_rand, label='Dados', c='k', alpha=0.8, s=15)
ax2 = ax.twinx()

# circle1 = plt.Circle((0, 0), 1, color='k', alpha=0.8, fill=False, lw=3)
# ax.add_patch(circle1)

for vals in [10000]:
#nuns = np.arange(0, 100, 1000)

    x_rand = [random.random() for _ in range(vals)]
    y_rand = [random.random() for _ in range(vals)]


    colors = []
    means = np.zeros(vals)
    var  = np.zeros(vals)
    for i in range(vals):
        x = x_rand[i]
        y = y_rand[i]
        if np.sqrt(x*x + y*y) > 1:
            colors.append(0)
        else:
            colors.append(1)
        means[i] = 4*np.mean(colors[:i])
        var[i]   = np.var(colors[:i])


    #plt.hist(means, label=f'est_{vals}', bins='auto', density=True)
    ax2.plot(var, label='variancia', c='r')
    ax.plot(means, label='valor estimado')
    print('vals', vals)
    ax.hlines(y=np.pi, xmax=vals, xmin=0, ls='--', colors='k', label=r'$\pi$')












#plt.xlim(3., 3.3)
ax.set_ylabel(r'media',  fontsize=25)
ax2.set_ylabel('variancia',  fontsize=25)
plt.xlabel('x', fontsize=25)
#ax.set_yticklabels(fontsize=25)
ax.tick_params(axis='y', labelsize=25)
ax2.tick_params(axis='y', labelsize=25)
#ax2.set_yticklabels(fontsize=25)
#plt.xticks(fontsize=25)
ax.legend(loc='lower right', fontsize=25)
ax2.legend(loc='upper right',fontsize=25)
plt.show()





#plt.plot(nuns, sigmoid(nuns), label=r'sigmoide', lw=2)
# plt.scatter(x_rand, y_rand, label=r'dados', c=colors, alpha=0.6)
# plt.xlabel('x')
# plt.ylabel('y', rotation=0)


# plt.ylim(0,1.0)
# plt.xlim(0,1.0)
# plt.legend()
# plt.show()