#!/usr/bin/env python
# coding: utf-8

#Import basic python librairies

import networkx as nx
import numpy as np
import pandas as pd
import random
from datetime import datetime
import numba
from numba import jit
import matplotlib.pyplot as plt

import igraph
import pylab
from igraph import *




# Function pour créer trade rates theta_ij ( powerlaw) 


######################################################### FIXED PARAMETERS
L = 5000 # number of herds

# Initialize network creation
theta_edges = create_edges_nbfils(L)

# Simulation setting
delta = 0.5
nb_steps = int(365/delta*3)

# Demographic parameters
demo_params = np.array([[1/(365*1.5), 1/(365*3)]]*L)
##########################################################################

















#########################################################################

#Save fixed parameters

#Save initial number of animals by herd
np.savetxt('N0s.txt', N0s)

#Save setting (delta and nb-steps)
setting = np.array([delta, nb_steps])
np.savetxt('setting.txt', setting)

#Save demo_params
np.savetxt('demo_params.txt', demo_params)

#Save theta_edges
np.savetxt('theta_edges.txt', theta_edges)

