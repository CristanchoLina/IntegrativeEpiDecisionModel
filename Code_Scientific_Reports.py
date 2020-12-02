#!/usr/bin/env python
# coding: utf-8

# In[Import modules]:


# Import modules
import numpy as np
import random
import numba
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import matplotlib.lines as mlines
# from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from datetime import datetime
import igraph
from collections import Counter
#from operator import itemgetter


# In[Set seed]:


# Set seed
np.random.seed(100)


# ### Functions definition #

# In[3]:


# Function for simulating a directed weighted powerlaw graph with weights = 1 all

def create_edges_nbfils(L, power, tries = 1000000):  
    
    # Generate de out-degree = in-degree sequence with the given power
    p= list(1 / (np.array(range(1, L)))**power) 
    p = p/sum(p)
    out_degs = list(np.random.choice(range(1, L), L, replace = True, p = p))
    
    # We correct the degree sequence if its sum is odd
    if (sum(out_degs) % 2 != 0):
        out_degs[0] = out_degs[0] + 1 
    
    # Generate directed graph with the given out-degree = in-degree sequence 
    g = igraph.Graph.Degree_Sequence(out_degs, out_degs, method="simple")
    g = g.simplify(multiple=True, loops=True) # remove loops or multiple edges
    
    print('Power:', power)
    g.es["weight"] = 1 # the graph is also weighted , the weights will later be modified

    edges = []
    weights = []
    for e in g.es:
        edges.append(e.tuple)
        weights.append(e["weight"])
    edges = np.array(edges)
    weights = np.array(weights)
    
    # Array with list f edges and weights. Columns: i,j,theta_ij
    theta_edges = np.hstack((edges, np.zeros((edges.shape[0], 1))))
    theta_edges[:,2] = weights 
    theta_edges = theta_edges.astype(float)
    theta_edges[:,2] = 1
    
    return np.array(theta_edges)


# In[4]:


# Function that converts theta_ij into probabilities
# Uses theta_edges, delta (fixed parameters)

def proba_edges(L, theta_edges, delta):
    
    proba_edges = []
    for k in range(0,L):
        theta_k = theta_edges[theta_edges[:,0] == k, 2]
        sum_theta_k = np.sum(theta_k)
        if sum_theta_k > 0:
            p_out_k = ((1.0 - np.exp(-sum_theta_k * delta))*theta_k/sum_theta_k)[:-1]
            p_out_k = np.append(p_out_k, 1 - np.sum(p_out_k))
        else:
            p_out_k = np.array([0.0])
        proba_edges.append([k, sum_theta_k, p_out_k])
        
    return np.array(proba_edges)


# ### Creation de fonctions pour l'épidemie  #

# In[5]:


# Function for creating the initial state of a herd with the given initial size and prop of infected animals

def create_initial_state(k, N0s, prop_infected_in_node):
    
    #Random generation of initial states for k
    N0 = N0s[k]
    I0 = int(N0*prop_infected_in_node)
    S0 = N0 - I0
    initial_state_k = [ S0,                                   # Snv
                        0,                                    # Snv to Inv
                        I0,                                   # Inv
                        0,                                    # Sv
                        0,                                    # Sv to Iv
                        0,                                    # Iv
                        0,                                    # I to R
                        0 ]                                   # R
    
    return np.array(initial_state_k)


# In[6]:


# Optimized function for drawing L multinomial samples, each with different probabilities
# for SIRstep_vectorized() function

@numba.jit(nopython=True)
def vec_multinomial(prob_matrix, sizes, res):
    for i in range(L):
        res[i] = np.random.multinomial(n = sizes[i], pvals = prob_matrix[i])
    return res 


# In[7]:


# Fonction of a SIR step given the current states of all herds:

def SIRstep_vectorized(current_states, simul):
    
    N = np.sum(current_states, axis = 1)
    
    Snv, Inv = current_states[:, 0],  current_states[:, 2]
    Sv, Iv = current_states[:, 3],  current_states[:, 5]
    R =  current_states[:, 7]

    betas_Inv = epid_params[:,0]
    taus = demo_params[:, 1]
    betas_Iv = betas_Inv * (1-eff_reduceinfec) 
    
    # Fixed epidemic probabilities
    p_B, p_I, p_R = fixed_epid_probas
    
    # Probabilities that change:
    
    # Probas for SNV
    lambds = (betas_Inv*(Inv)/N) + (betas_Iv*(Iv)/N) 
    lambds[np.isnan(lambds)] = 0.

    Snv_rates = lambds + taus + thetas
    p_SnvInv = (1.0 - np.exp(-Snv_rates* delta))*lambds/Snv_rates 
    p_SnvD = (1.0 - np.exp(-Snv_rates * delta))*taus/Snv_rates
    p_Snvout = (1.0 - np.exp(-Snv_rates * delta))*thetas/Snv_rates

    # Probas for SV
    lambds_v = (1-eff_protect) * ( (betas_Inv*(Inv)/N) + (betas_Iv*(Iv)/N)) 
    lambds_v [np.isnan(lambds_v)] = 0.

    Sv_rates = lambds_v + taus + thetas
    p_SvIv = (1.0 - np.exp(-Sv_rates * delta))*lambds_v/Sv_rates
    p_SvD = (1.0 - np.exp(-Sv_rates * delta))*taus/Sv_rates
    p_Svout = (1.0 - np.exp(-Sv_rates * delta))*thetas/Sv_rates 

    #Add the probabilities
    p_Snv = np.array([p_SnvInv, p_SnvD, p_Snvout, 1.0-(p_SnvInv + p_SnvD + p_Snvout)]).T 
    p_Sv = np.array([p_SvIv, p_SvD, p_Svout, 1.0-(p_SvIv + p_SvD + p_Svout)]).T 
    
    # Draw from multinomials for each compartment:
    B_sample = vec_multinomial(prob_matrix = p_B, sizes = N.astype(int), res = np.zeros(shape=(L,2)))
    Snv_sample = vec_multinomial(prob_matrix = p_Snv, sizes = Snv.astype(int), res = np.zeros(shape=(L,4)))
    Sv_sample = vec_multinomial(prob_matrix = p_Sv, sizes = Sv.astype(int), res = np.zeros(shape=(L,4)))
    Inv_sample = vec_multinomial(prob_matrix = p_I, sizes = Inv.astype(int), res = np.zeros(shape=(L,4)))
    Iv_sample = vec_multinomial(prob_matrix = p_I, sizes = Iv.astype(int),res = np.zeros(shape=(L,4)))
    R_sample = vec_multinomial(prob_matrix = p_R, sizes = R.astype(int), res = np.zeros(shape=(L,3)))
    
    #Add samples and update counts in compartments:
    d_SnvI, d_SvI, d_InvR, d_IvR = Snv_sample[:, 0], Sv_sample[:,0], Inv_sample[:,0], Iv_sample[:,0]
    births =  B_sample[:,0] 
    conditioned_births = births*(capacities - N > 0) # Actual births are limited by herd capacity 
    Snv = Snv_sample[:,3] + conditioned_births
    Sv = Sv_sample[:,3]
    Inv = Inv_sample[:,3] + d_SnvI
    Iv = Iv_sample[:,3] + d_SvI
    R = R_sample[:,2] + d_InvR + d_IvR
    Snv_out, Inv_out, Sv_out, Iv_out = Snv_sample[:,2], Inv_sample[:,2], Sv_sample[:,2], Iv_sample[:,2]
    R_out = R_sample[:,1]
    
    # Return list of two arrays: current state, and exports by compartment.
    return np.array([Snv, d_SnvI, Inv, Sv, d_SvI, Iv, d_InvR + d_IvR, R]).T.astype(int),           np.array([Snv_out, Inv_out, Sv_out, Iv_out, R_out]).T.astype(int)


# In[8]:


# Optimized fonction for assigning exports 

@numba.jit(nopython=True)
def vec_exports_i(out_k, p_out_k):
    nb_neighb = len(p_out_k)
    res_k = np.zeros((5,nb_neighb))
    for i in range(5):
        res_k[i] = np.random.multinomial(out_k[i], p_out_k)
    return res_k.T


def vec_exports(thetas, probs_exports, outs):
    res = []
    for k in range(L):
        theta_k, p_out_k = thetas[k], probs_exports[k]
        if theta_k != 0:
            res_k = vec_exports_i(outs[k], p_out_k)
            res.append(res_k)
    return res


# In[9]:


# Functions for defining the decision mechanism
# Used later in the simulator

def nothing(*args):
    return np.zeros(L)

def always(*args):
    return np.ones(L)

def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


def log_exp_Sum(log_weights):
    exp_logw = []
    for log_w in log_weights:
        exp_logw.append(np.exp(log_w))
    return np.log(sum(exp_logw))

def expw(simul, L, mean_rewards, counts, vaccinators, non_vaccinators, relat_reward,decision_times,
         log_weights, kappas, *args):
    
    #update log weights
    log_weights[vaccinators, 1] += relat_reward[vaccinators] * kappas[vaccinators]
    log_weights[non_vaccinators, 0] += relat_reward[non_vaccinators] * kappas[non_vaccinators]
    
    decisions = np.zeros(L)
    for k in range(0,L):
        log_exp_Sum_k = np.logaddexp(log_weights[k][0], log_weights[k][1])  #float(log_exp_Sum(log_weights[k]))
        probabilityDistribution_k = tuple((np.exp(w - log_exp_Sum_k)) for w in log_weights[k])
        decisions[k] = draw(probabilityDistribution_k)
        
    return decisions

def neighb_expw(simul, L, mean_rewards, counts, vaccinators, non_vaccinators, relat_reward, decision_times,
         log_weights, kappas, rhos, prev_decision, theta_edges_compact, *args):
        
    for k in range(0,L):
        # List of neighbors of k:
        neighbors_k = theta_edges_compact[theta_edges_compact[:, 0] == k, 1][0]
        if neighbors_k != []:
            #Choose a random neighbor
            neighbor = np.random.choice(neighbors_k)
            neigh_prev_decision = prev_decision[neighbor]
            neigh_reward = relat_reward[neighbor]
            log_weights[k][neigh_prev_decision] += neigh_reward* rhos[k]
    
    #update log weights
    log_weights[vaccinators, 1] += relat_reward[vaccinators] * kappas[vaccinators]
    log_weights[non_vaccinators, 0] += relat_reward[non_vaccinators] * kappas[non_vaccinators]
    
    decisions = np.zeros(L)
    for k in range(0,L):
        log_exp_Sum_k = np.logaddexp(log_weights[k][0], log_weights[k][1]) #float(log_exp_Sum(log_weights[k]))
        probabilityDistribution_k = tuple((np.exp(w - log_exp_Sum_k)) for w in log_weights[k])
        decisions[k] = draw(probabilityDistribution_k)
        
    return decisions


# In[10]:


# Optimized function for applying the decision: SNV pass to SV, or SV pass to SNV

@numba.jit(nopython=True)
def vaccinate(current_states, vacc):
    
    vaccinators = np.where(vacc == 1)[0]
    non_vaccinators = np.where(vacc != 1)[0]
    
    states_vaccinators = np.copy(current_states[vaccinators,:])
    current_states[vaccinators,3] = states_vaccinators[:,0] + states_vaccinators[:,3]
    current_states[vaccinators,0] = np.zeros(len(vaccinators))
    
    states_non_vaccinators = np.copy(current_states[non_vaccinators,:])
    current_states[non_vaccinators,0] = states_non_vaccinators[:,0] + states_non_vaccinators[:,3]
    current_states[non_vaccinators,3] = np.zeros(len(non_vaccinators))
        
    return current_states


# In[11]:


# Fonction for epidemic-decision path for all herds

def path(initial_states, demo_params, epid_params, eco_params, fixed_epid_probas,
         neighbors_list, parents_list, probs_exports, duration_decision, eff_reduceinfec, eff_protect,
         thetas, delta, nb_steps, nexpw_params, theta_edges_compact,
         mechanism = 'neighb_expw'):
    
    #Initialization
    all_states = np.zeros((nb_steps, L, 8), dtype=int) 
    all_states[0] = np.copy(initial_states) #Snv, SnvI, Inv , Sv, SvI, Iv , IR, R
    ventes_byfarm = np.zeros((nb_steps, L))
    achats_byfarm = np.zeros((nb_steps, L))
    
    # Economic costs
    r, phi, cu_vacc, cf_vacc = eco_params
    c_inf = phi*r
    
    #Decision times(fct of nb of steps and duration decision)
    simul_list = np.array(range(0, nb_steps))
    decision_times = simul_list[np.mod(simul_list*delta, duration_decision) == 0.0] + 1
    decision_times = decision_times[1:]
    decisions = np.zeros((nb_steps, L), dtype=int)
    
    #For expw strategy 
    
    sizes = np.zeros((nb_steps, L))
    counts = np.array([[0., 0.]] * L )
    mean_rewards = np.array([[0., 0.]] * L) 
    relat_reward = np.zeros(L)
    vaccinators, non_vaccinators = [], []
        
    init_proba_vacc, kappa, rho = nexpw_params
    kappas = np.array([kappa] * L) 
    
    #Convert probas to weights
    if init_proba_vacc == 1.:
        w_nv = 0.
        w_v = 1.   
    else: 
        w_v_w_nv = init_proba_vacc/(1.-init_proba_vacc)
        w_nv = 2.
        w_v = w_nv*w_v_w_nv
    
    log_weights = np.array([[np.log(w_nv), np.log(w_v)]] * L) #prob de pas vacc, prob de vacc
    
    #For neighb_expw strategy 
    rhos = np.array([rho] * L )
    prev_decision = np.zeros(L, dtype=int)
    
    #Choose mechanism
    if mechanism == 'nothing':    
        decision_function = nothing
    elif mechanism == 'always':    
        decision_function = always
    elif mechanism == 'expw':    
        decision_function = expw
    elif mechanism == 'neighb_expw':    
        decision_function = neighb_expw
        
    #Evolution du path
    
    for simul in range(1, nb_steps):
        
        current_states = np.copy(all_states[simul-1])
        sizes[simul-1] = np.sum(np.delete(current_states, (1, 4, 6), 1), axis = 1)
        
        # Decision if simul is a decision moment
        if simul in decision_times:
            
            #Compute reward of previous decision
            if simul != decision_times[0]:
                
                time_prev_decision = decision_times[np.where(decision_times == simul)[0][0] - 1]
                prev_decision = decisions[time_prev_decision]
                N_prev_decision = sizes[time_prev_decision]
                Nt_sum = np.sum(sizes[time_prev_decision:simul], axis = 0)
                nb_newinf = np.sum(np.sum(all_states[time_prev_decision:simul], axis = 0)[:,(1,4)], axis = 1)
    
                nb_newinf = np.maximum(nb_newinf, 0)
                reward_dec = -(cf_vacc + (cu_vacc*N_prev_decision))*prev_decision - (c_inf*nb_newinf)  
                relat_reward = reward_dec/Nt_sum #np.divide(reward_dec, N_prev_decision, out=np.zeros_like(reward_dec), where=N_prev_decision!=0)
                vaccinators = np.where(prev_decision == 1.)
                non_vaccinators = np.where(prev_decision == 0.)
                
            #Take decision
            decisions[simul] = decision_function(simul, L, mean_rewards, counts, vaccinators, non_vaccinators,
                                                 relat_reward, decision_times, log_weights, kappas, rhos, prev_decision, theta_edges_compact)
            weights = np.exp(log_weights)
            probas = np.round(weights / np.sum(weights, axis = 1)[:,None], 2)
            prob_vacc = probas[:,1]
            prob_vacc   = prob_vacc[prob_vacc  != 1.]
            # print(stats.mode(prob_vacc))
            # print(np.mean(prob_vacc))
            # print(prob_vacc)
            
            #Record decisions
            decisions_simul = decisions[simul]
            
            #Decisions are applied here 
            current_states = vaccinate(current_states, decisions_simul)
    
        #Change states
        prev_N = np.sum(current_states, axis = 1)
        current_states, outs = SIRstep_vectorized(current_states, simul)
        ventes_byfarm[simul] = np.sum(outs, axis = 1)
        
        #Assign exports
        exports = np.concatenate(vec_exports(thetas, probs_exports, outs))
    
        #Assign exports as imports
        
        open_neighbors_indicator = ((capacities- prev_N)[neighbors_list] > 0) 
        
        imports =[]
        returns = []
        for c in range(0, 5):
            souhait = ((capacities- prev_N)[neighbors_list])
            weights = open_neighbors_indicator* list(map(min, exports[:,c], souhait))
            
            unsold = exports[:,c] - weights
            imports.append(np.bincount(neighbors_list, weights=weights))
            returns.append(np.bincount(parents_list, weights=unsold))
        
        imports = np.array(imports).T 
        modif_imports = np.insert(imports, [1,3,4], 0, axis=1)
        
        returns = np.array(returns).T
        modif_returns = np.insert(returns , [1,3,4], 0, axis=1)
            
        all_states[simul] = current_states + modif_imports + modif_returns
                           
        achats_byfarm[simul] = np.sum(modif_imports, axis = 1)
        ventes_byfarm[simul] = ventes_byfarm[simul] - np.sum(modif_returns, axis = 1)
        
    return decision_times, decisions, all_states, ventes_byfarm, achats_byfarm


# # Fix setting 

# In[12]:


# Total number of herds
L = 5000 

# Fixed graph structure with the given porwer-law (weights will be defined later)
power = 2
theta_edges = create_edges_nbfils(L, power)

# Simulation setting
delta = 0.5 # simulation step
nb_years = 3 # number of years to simulate
nb_steps = int(365/delta*nb_years) # length of each trajectory

# Demographic parameters: mu (birth rate) and tau (death rate)
demo_params = np.array([[1/(365*1.5), 1/(365*3)]]*L) 

print('Number of herds:', L)
print('Simulation step delta:', delta)
print('Simulated years:', nb_steps*delta/365)
print('Demographic parameters (mu and tau):', demo_params[0])


# In[13]:


# We add a neighbor for herds without buyers, or sellers 
# if the neighbor was itself, maybe it has no neighbor

def find_missing(lst): 
    return [x for x in range(0, L)
                               if x not in lst] 

sorted_receveirs =  sorted(theta_edges[:,1].astype(int))
non_receveirs = find_missing(sorted_receveirs)
theta_edges = list(theta_edges)
for i in non_receveirs:
    if i == 0:
        theta_edges.append(np.array([i+1, i, 1]))
    else:
        theta_edges.append(np.array([i-1, i, 1])) 
theta_edges = np.array(theta_edges)

sorted_givers =  sorted(theta_edges[:,0].astype(int))
non_givers = find_missing(sorted_givers)
theta_edges = list(theta_edges)
for i in non_givers:
    if i == 0:
        theta_edges.append(np.array([i, i+1, 1]))
    else:
        theta_edges.append(np.array([i, i-1, 1])) 
theta_edges = np.array(theta_edges)


print('Aditionally created in-edges:', len(non_receveirs))
print('Additionally created out-edges:', len(non_givers))


# In[14]:


# Plot histogram of in_degree and out-degree , simple and in log

a = []
b = []
for i in range(L):
    a.append(np.sum(theta_edges[:,1] == i))
    b.append(np.sum(theta_edges[:,0] == i))
    
in_deg = np.array(a)
out_deg = np.array(b)
in_deg_pd = pd.DataFrame(in_deg)
out_deg_pd = pd.DataFrame(out_deg)

plt.figure()
in_deg_pd.hist(bins = 100,  weights=np.zeros_like(in_deg_pd) + 1. / in_deg_pd.size)
plt.tight_layout()
plt.title('Distribution in(-out) degree')
plt.show()



degrees = in_deg
degree_counts = Counter(degrees)                                                                                                 
x, y = zip(*degree_counts.items())                                                      

plt.figure(1)   

# prep axes                                                                                                                      
plt.xlabel('log(degree)')                                                                                                             
plt.xscale('log')           

plt.ylabel('log(frequency)')                                                                                                          
plt.yscale('log')                                                                                                                   
                                                                                                                                     # do plot                                                                                                                        
plt.scatter(x, y, marker='.')                                                                                                    
plt.show()


# In[15]:


# Initial size herds

N0s = np.random.gamma(9,12, L)
N0s = N0s.astype(int)
N0s_pd = pd.DataFrame(N0s)

#%%
import seaborn as sns; sns.set(style="ticks", color_codes=True)
plt.figure()
N0s_pd.hist(bins =35,  weights=np.zeros_like(N0s_pd) + 1. / N0s_pd.size)
plt.title('')
plt.xlabel('initial herd size')
plt.ylabel('frequency')
plt.show()


print('Range initial size:', '(', min(N0s), ',', max(N0s), ')')


# In[16]:


# Assign sizes according to out degree:

#sort thetas_i from small to big
df_out_degrees = pd.DataFrame(out_deg)
df_out_degrees['indegree']  = in_deg # add indeg to the database too
N0s_pd  = N0s_pd.sort_values(0) 
sorted_bygroup_N0s = np.array(N0s_pd[0])

# Data frame de degrees avec N0s
df_out_degrees  = df_out_degrees.sort_values(0)
df_out_degrees['N0s'] = sorted_bygroup_N0s
df_out_degrees = df_out_degrees.sort_index()


# In[17]:


# Set maximal capacities
max_cap = 1.5
N0s = np.array(df_out_degrees['N0s'])
capacities = N0s*max_cap


# In[18]:


# Simulate out rates theta_i

p=list(1 / (np.array(np.arange(0.0006,  1, 0.000001)))**power) 
p = p/sum(p)
out_thetas = list(np.random.choice(np.arange(0.0006, 1,  0.000001), L, replace = True, p = p)) 
out_thetas = pd.DataFrame(out_thetas)
plt.figure()
out_thetas.hist(bins = 100,  weights=np.zeros_like(out_thetas) + 1. / out_thetas.size)
plt.tight_layout()
plt.title('Distribution du taux de sortie')
plt.show()

#sort thetas_i from small to big
out_thetas  = out_thetas.sort_values(0)

sorted_bygroup_thetas_i = np.array(out_thetas[0])


# In[19]:


# Assign theta_i according to out-degree

df_out_degrees  = df_out_degrees.sort_values(0)
df_out_degrees['theta_i'] = sorted_bygroup_thetas_i
df_out_degrees = df_out_degrees.sort_index()


# In[20]:


# Correlation between N0, theta_i and out_degree
print('Spearman correlation:')
print(df_out_degrees.corr(method='spearman'))
print('Pearson correlation')
print(df_out_degrees.corr(method='pearson'))


# In[21]:


# Distribute theta_i among child nodes (buyers) to obtain the theta_ij

for i in range(0,L):
    ijw = theta_edges[theta_edges[:,0] == i, :] 
    neighb_i = ijw[:,1].astype(int)
    theta_i_out = np.array(df_out_degrees['theta_i'])[i]
    outdeg_neighi = out_deg[neighb_i]
    indeg_neighi = in_deg[neighb_i]
    sizes_neighi = N0s[neighb_i]
    theta_neighi_out = np.array(df_out_degrees['theta_i'])[tuple([neighb_i])]
    theta_prime = 1/indeg_neighi # inversely proportional to the in-degree 
    theta_i_neighi = theta_prime * theta_i_out / np.sum(theta_prime)
    theta_edges[theta_edges[:,0] == i, 2] = theta_i_neighi

theta_pd = pd.DataFrame(theta_edges)


# ### Particular setting of not fixed parameters in these simulations

# In[22]:


# Initial conditions
prop_inf_nodes = 0.1 # proportion of initially infected nodes 
prop_inf_in_node = 0.15 # proportion of animals infected in the infected nodes 

# Epidemic parameters
recovery_time = 90
R0 = 2
gamma = 1/recovery_time
beta = R0*gamma
epid_params = np.array([[beta, gamma]]*L)


# Control measures parameters
eff_protect = 1
eff_reduceinfec = 0

# Economic parameters
r = 2000
phi = 0.8
cf_vacc = 50
cu_vacc = 5

eco_params = np.array([r, phi, cu_vacc, cf_vacc])

# Decision parameters
duration_decision = 180

#Neigh expw specific parameters
init_proba_vacc = 0.01
kappa = 1
rapport_rho_kappa = 0.5
rho = rapport_rho_kappa*kappa

nexpw_params = np.array([init_proba_vacc, kappa, rho])


# In[23]:


print('Prop. of initially infected nodes: ', prop_inf_nodes)
print('Prop. infected animals in the initially infected nodes: ', prop_inf_in_node)
print('Infetious period (1/gamma): ', recovery_time)
print('Beta/Gamma: ', R0)
print('Beta and Gamma: ', epid_params[0])
print('Value of a healthy animal (r): ', r)
print('Loss of value of an infected animal (phi): ', phi)
print('Protection efficacy of the vaccin: ', eff_protect)
print('Fixed cost of vaccine: ', cf_vacc)
print('Unit cost of vaccine: ', cu_vacc)
print('Duration of vaccine protection (and of decision): ', duration_decision)
print('Initial probability of vaccinating:', init_proba_vacc)
print('kappa (Farmer\'s sensitivity to own results): ', kappa)
print('rho (Breeder\'s sensitivity to neighbor\'s results): ', rho)


# In[24]:


#Fixed useful arrays

#thetas et probas de sortie (fct de matrix theta_edges)
probs_edges = proba_edges(L, theta_edges, delta) # k, sum_theta_k, p_out_k (vector)

#thetas
mus, taus = demo_params[:,0], demo_params[:,1]
thetas = probs_edges[:,1].astype(float)

#list of prob. of exit
probs_exports = list(probs_edges[:,2])

# list of childs of each herd (for imports later)
neighbors_list = theta_edges[:,1].astype(int)

# list of parents of each herd (for returns later)
parents_list = theta_edges[:, 0].astype(int)

#theta edges in format for neighbor ewpw strategy

theta_edges_compact = []
for k in range(0,L):
    neighbors_k = []
    for w in theta_edges[theta_edges[:,0]== k]:
        neighbors_k.append(int(w[1]))
    for w in theta_edges[theta_edges[:,1]== k]:
        neighbors_k.append(int(w[0]))
    theta_edges_compact.append([k, neighbors_k ])
theta_edges_compact = np.array(theta_edges_compact)


# In[25]:


#Fixed probabilities

# prob. of birth
p_B = 1.0 - np.exp(-mus * delta)

#prob. for R 
R_rates = taus + thetas
p_RD=  (1.0 - np.exp(- R_rates * delta))*taus/ R_rates 
p_Rout =  (1.0 - np.exp(-R_rates * delta))*thetas/R_rates 

#prob. for I
I_rates = gamma + R_rates
p_IR =  (1.0 - np.exp(-I_rates * delta))*gamma/I_rates
p_ID =  (1.0 - np.exp(-I_rates * delta))*taus/I_rates
p_Iout = (1.0 - np.exp(-I_rates * delta))*thetas/I_rates

#Stock prob. vectors
p_B = np.array([p_B, 1.0-p_B]).T 
p_I = np.array([p_IR, p_ID, p_Iout, 1.0-(p_IR + p_ID + p_Iout)]).T 
p_R = np.array([p_RD, p_Rout, 1.0-(p_RD + p_Rout)]).T

#fixed_epid_probas 
fixed_epid_probas = [p_B, p_I, p_R]


# # Number of runs

# In[26]:


simulations = 1


# In[27]:


# Initial state creation 
    
all_initial_states = []

for i in range(0, simulations):
    
    random.seed(i)

    perm = random.sample(range(L), L)
    num_infected_noeuds = int(prop_inf_nodes*L)

    initial_states_inf = []
    initial_states_noninf = [] 
    for k in perm[:num_infected_noeuds]:
        initial_states_inf.append([k, create_initial_state(k, N0s, prop_inf_in_node)])
    for k in perm[num_infected_noeuds:]:
        initial_states_noninf.append([k, create_initial_state(k, N0s, 0)])
    initial_states = initial_states_inf + initial_states_noninf
    initial_states = sorted(initial_states, key=lambda index: index[0])
    initial_states = np.stack(np.array(initial_states)[:, 1])
    all_initial_states.append(initial_states)
    
    all_initial_states.append(initial_states)


# # C. Neighbor-exp strategy ($\kappa = 1, \rho = 0.5$ )

# In[30]:


#Action

action_results = np.zeros((simulations, nb_steps, L, 8))
ventes_action = np.zeros((simulations, nb_steps, L))
achats_action = np.zeros((simulations, nb_steps, L))
action_behaviors = np.zeros((simulations, nb_steps, L))
    
start_time = datetime.now()
for i in range(0, simulations):
    
    decision_times_i, behavior_i, results_i, ventes_i, achats_i = path(all_initial_states[i], demo_params, epid_params, eco_params, fixed_epid_probas,
         neighbors_list, parents_list, probs_exports, duration_decision, eff_reduceinfec, eff_protect,
         thetas, delta, nb_steps, nexpw_params, theta_edges_compact, mechanism = 'neighb_expw')
    sir_i = np.zeros((nb_steps, L, 5))
    sir_i[:,:,0] = results_i[:,:,0] + results_i[:,:,3] #S
    sir_i[:,:,1] = results_i[:,:,1] + results_i[:,:,4] #S to I
    sir_i[:,:,2] = results_i[:,:,2] + results_i[:,:,5] #I
    sir_i[:,:,3] = results_i[:,:,6]  #I to R
    sir_i[:,:,4] = results_i[:,:,7]  # R
    action_results[i] = results_i
    ventes_action[i] = ventes_i 
    achats_action[i] = achats_i
    action_behaviors[i] = behavior_i
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


#%%
# # D. Neighbor-exp strategy ($\kappa = 25, \rho = 12.5$ )

# In[31]:
    
kappa = 25
rapport_rho_kappa = 0.5
rho = rapport_rho_kappa*kappa

nexpw_params = np.array([init_proba_vacc, kappa, rho])


# In[32]:


#Action

action25_results = np.zeros((simulations, nb_steps, L, 8))
action25_action = np.zeros((simulations, nb_steps, L))
action25_behaviors = np.zeros((simulations, nb_steps, L))
ventes_action25 = np.zeros((simulations, nb_steps, L))
achats_action25 = np.zeros((simulations, nb_steps, L))
    
start_time = datetime.now()
for i in range(0, simulations):
    
    decision_times_i, behavior_i, results_i, ventes_i, achats_i = path(all_initial_states[i], demo_params, epid_params, eco_params, fixed_epid_probas,
         neighbors_list, parents_list, probs_exports, duration_decision, eff_reduceinfec, eff_protect,
         thetas, delta, nb_steps, nexpw_params, theta_edges_compact, mechanism = 'neighb_expw')
    sir_i = np.zeros((nb_steps, L, 5))
    sir_i[:,:,0] = results_i[:,:,0] + results_i[:,:,3] #S
    sir_i[:,:,1] = results_i[:,:,1] + results_i[:,:,4] #S to I
    sir_i[:,:,2] = results_i[:,:,2] + results_i[:,:,5] #I
    sir_i[:,:,3] = results_i[:,:,6]  #I to R
    sir_i[:,:,4] = results_i[:,:,7]  # R
    action25_results[i] = results_i
    ventes_action25[i] = ventes_i 
    achats_action25[i] = achats_i
    action25_behaviors[i] = behavior_i
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# # Plots

# In[35]:


# Action
sir_r = np.zeros((simulations, nb_steps, L, 3))
sir_r[:,:,:,0] = action_results[:,:,:,0] +  action_results[:,:,:,3] #S
sir_r[:, :,:,1] = action_results[:, :,:,2] + action_results[:, :,:,5]  #I
sir_r[:, :,:,2] = action_results[:, :,:,7]  # R
N_action = np.sum(sir_r, axis = 3)
I_action = action_results[:,:,:,2]

# Action 25
sir_r = np.zeros((simulations, nb_steps, L, 3))
sir_r[:,:,:,0] = action25_results[:,:,:,0] +  action25_results[:, :,:,3] #S
sir_r[:, :,:,1] = action25_results[:, :,:,2] +action25_results[:, :,:,5]  #I
sir_r[:, :,:,2] = action25_results[:, :,:,7]  # R
N_action25 = np.sum(sir_r, axis = 3)
I_action25 = action25_results[:,:,:,2]


# In[38]:
#Proportion of infected herds
I_action = action_results[:,:,:,2]
prop_infected_farms_time_action  = np.sum(I_action!=0, axis = 2)/L
mean_prop_infected_farms_time_action = pd.DataFrame(np.mean(prop_infected_farms_time_action, axis = 0))
upp_prop_infected_farms_time_action = pd.DataFrame(np.percentile(prop_infected_farms_time_action, 90, axis = 0))
low_prop_infected_farms_time_action =  pd.DataFrame(np.percentile(prop_infected_farms_time_action, 10, axis = 0))


I_action25 = action25_results[:,:,:,2]
prop_infected_farms_time_action25   = np.sum(I_action25 !=0, axis = 2)/L
mean_prop_infected_farms_time_action25  = pd.DataFrame(np.mean(prop_infected_farms_time_action25 , axis = 0))
upp_prop_infected_farms_time_action25  = pd.DataFrame(np.percentile(prop_infected_farms_time_action25 , 90, axis = 0))
low_prop_infected_farms_time_action25  =  pd.DataFrame(np.percentile(prop_infected_farms_time_action25 , 10, axis = 0))

#Proportion of herds that vaccinate

prop_vaccinators = np.mean(action_behaviors,axis = 2)[:, decision_times_i]
mean_prop_vaccinators = np.mean(prop_vaccinators, axis = 0)
pd_mean_prop_vaccinators = pd.DataFrame(mean_prop_vaccinators, decision_times_i) #mean_prop_vaccinators
low_prop_vaccinators = mean_prop_vaccinators -np.percentile(prop_vaccinators, 10, axis = 0) 
upp_prop_vaccinators = np.percentile(prop_vaccinators, 90, axis = 0) - mean_prop_vaccinators

prop25_vaccinators = np.mean(action25_behaviors,axis = 2)[:, decision_times_i]
mean_prop25_vaccinators = np.mean(prop25_vaccinators, axis = 0)
pd_mean_prop25_vaccinators = pd.DataFrame(mean_prop25_vaccinators, decision_times_i)
low_prop25_vaccinators = mean_prop25_vaccinators -np.percentile(prop25_vaccinators, 90, axis = 0) 
upp_prop25_vaccinators = np.percentile(prop25_vaccinators, 10, axis = 0) - mean_prop25_vaccinators

a = np.array([low_prop_vaccinators, upp_prop_vaccinators])
b = np.array([low_prop25_vaccinators, upp_prop25_vaccinators])


#%%

plt.rcParams.update({'font.size': 10})
f, ax = plt.subplots(1)
for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(1)
plt.style.use("default")
#plt.ylim([-0.05,0.55])
#plt.xlim([-60.,1095+60])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

xpoints = decision_times_i

for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'lightgray')

days = list(range(nb_steps))

action_line = plt.plot(mean_prop_infected_farms_time_action, linewidth = 1.5, color= '#fdae61')
upp_action_line = plt.plot(upp_prop_infected_farms_time_action ,  linestyle=':',  linewidth = 1.5,color= '#fdae61' )
low_action_line = plt.plot(low_prop_infected_farms_time_action ,  linestyle=':', linewidth =1.5,color= '#fdae61' )

action25_line = plt.plot(mean_prop_infected_farms_time_action25,  linewidth = 1.5, color= '#74add1')
upp_action25_line = plt.plot(upp_prop_infected_farms_time_action25 ,  linestyle=':', linewidth = 1.5, color = '#74add1')
low_action25_line = plt.plot(low_prop_infected_farms_time_action25 ,  linestyle=':', linewidth = 1.5, color = '#74add1')

action = mlines.Line2D([], [], color='#fdae61', linewidth=1.5, label='neigh-expw (1)')
action25 = mlines.Line2D([], [], color='#74add1', linewidth=1.5, label='neigh-expw (25)')

#plt.title('Proportion of infected herds', fontweight = "bold")
plt.xlabel("day", fontsize = 10)
plt.ylabel("proportion of infected herds", fontsize = 10)
plt.errorbar(xpoints, np.array(pd_mean_prop_vaccinators), yerr=a, fmt = 'o', color = '#fdae61',linewidth = 1.5)
plt.errorbar(xpoints, np.array(pd_mean_prop25_vaccinators), yerr=b, fmt = 'o', color = '#74add1',linewidth = 1.5)

#plt.errorbar(xpoints, np.array(pd_mean_prop_vaccinators.iloc[0]), yerr=a[0], fmt = 'o', color = '#1a9641')
#plt.errorbar(xpoints, np.array(pd_mean_prop25_vaccinators.iloc[0]), yerr=b[0], fmt = 'o', color = 'darkorange')

fontP = FontProperties()
fontP.set_size(9.5)
legend = plt.legend(handles = [action, action25], prop=fontP)
frame = legend.get_frame()

#%%
# Proportion of infected animals among infected herds

#rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Tahoma']

Iaction_nonzeros = np.zeros((simulations, nb_steps, 3))
Iaction25_nonzeros = np.zeros((simulations, nb_steps, 3))

for s in range(simulations):
    for t in range(0,nb_steps):
        I_t = I_action[s, t]/N_action[s,t]
        if I_t[I_t!= 0.0].size:
            Iaction_nonzeros[s, t, 0] = np.nanmean(I_t[I_t!= 0.0]) 
            Iaction_nonzeros[s, t, 1] = np.nanpercentile(I_t[I_t!= 0.0], 90)
            Iaction_nonzeros[s, t, 2] = np.nanpercentile(I_t[I_t!= 0.0], 10)
        else:
            Iaction_nonzeros[s, t, 0] = 0
            Iaction_nonzeros[s, t, 1] = 0
            Iaction_nonzeros[s, t, 2] = 0

        
Iaction_pd_nonzeros = pd.DataFrame(Iaction_nonzeros[:, :, 0].T)
UC_Iaction_pd_nonzeros = pd.DataFrame(Iaction_nonzeros[:, :, 1].T)
LC_Iaction_pd_nonzeros = pd.DataFrame(Iaction_nonzeros[:, :, 2].T)

for s in range(simulations):
    for t in range(0,nb_steps):
        I_t = I_action25[s, t]/N_action25[s,t]
        if I_t[I_t!= 0.0].size:
            Iaction25_nonzeros[s, t, 0] = np.nanmean(I_t[I_t!= 0.0]) 
            Iaction25_nonzeros[s, t, 1] = np.nanpercentile(I_t[I_t!= 0.0], 90)
            Iaction25_nonzeros[s, t, 2] = np.nanpercentile(I_t[I_t!= 0.0], 10)
        else:
            Iaction25_nonzeros[s, t, 0] = 0
            Iaction25_nonzeros[s, t, 1] = 0
            Iaction25_nonzeros[s, t, 2] = 0

Iaction25_pd_nonzeros = pd.DataFrame(Iaction25_nonzeros[:, :, 0].T)
UC_Iaction25_pd_nonzeros = pd.DataFrame(Iaction25_nonzeros[:, :, 1].T)
LC_Iaction25_pd_nonzeros = pd.DataFrame(Iaction25_nonzeros[:, :, 2].T)

# In[39]:


prop_intra_action_mean = pd.DataFrame(np.mean(Iaction_pd_nonzeros, axis = 1))
prop_intra_action_upp = pd.DataFrame(np.percentile(UC_Iaction_pd_nonzeros, axis = 1, q = 90))
prop_intra_action_low = pd.DataFrame(np.percentile(LC_Iaction_pd_nonzeros, axis = 1, q =10))

prop_intra_action25_mean = pd.DataFrame(np.mean(Iaction25_pd_nonzeros, axis = 1))
prop_intra_action25_upp = pd.DataFrame(np.percentile(UC_Iaction25_pd_nonzeros, axis = 1,q =90))
prop_intra_action25_low = pd.DataFrame(np.percentile(LC_Iaction25_pd_nonzeros, axis = 1,q =10))


# In[40]:


plt.rcParams.update({'font.size': 10})
f, ax = plt.subplots(1)
for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
    ax.spines[side].set_linewidth(1)
plt.style.use("default")
#plt.ylim([-0.02,0.38])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

xpoints = decision_times_i

for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'lightgray')

days = list(range(nb_steps))


upp_action_line = plt.plot(prop_intra_action_upp, linewidth = 1.5, color = '#fdae61', label = 'Q90',
                           linestyle=':')
low_action_line = plt.plot(prop_intra_action_low, linewidth = 1.5, color = '#fdae61', label = 'Q10',
                          linestyle=':')

mean_action_line = plt.plot(prop_intra_action_mean, linewidth = 1.5, color = '#fdae61', label = 'action')



upp_action25_line = plt.plot(prop_intra_action25_upp, linewidth = 1.5, color = '#74add1', label = 'Q90',
                           linestyle=':')
low_action25_line = plt.plot(prop_intra_action25_low, linewidth = 1.5, color = '#74add1', label = 'Q10',
                          linestyle=':')
mean_action25_line = plt.plot(prop_intra_action25_mean, linewidth = 1.5, color = '#74add1', label = 'action25')


action = mlines.Line2D([], [], color='#fdae61', linewidth=1.5, label='neigh-expw (1)')
action25 = mlines.Line2D([], [], color='#74add1', linewidth=1.5, label='neigh-expw (25)')


#plt.title('Proportion of infected animals in an infected herd', fontweight = "bold")
plt.xlabel("day", fontsize = 10)
plt.ylabel("intra-herd proportion of infected animals", fontsize = 10)
frame.set_facecolor("white")
frame.set_linewidth(0)

from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size(9.5)
legend = plt.legend(handles = [action, action25], prop=fontP)
frame = legend.get_frame()



# In[41]:

# Mean Herd sizes through time
N = np.sum(sir_r, axis = 3)
mean_sizes_nopd = np.mean(N, axis = 0)
mean_sizes = pd.DataFrame(mean_sizes_nopd)

# Final mean herd size distribution

N_T = mean_sizes_nopd[nb_steps-1]
N_T_pd = pd.DataFrame(N_T)

import seaborn as sns; sns.set(style="ticks", color_codes=True)
plt.figure()
N_T_pd.hist(bins =35,  weights=np.zeros_like(N_T_pd) + 1. / N_T_pd.size)
plt.title('')
plt.xlabel('final herd size')
plt.ylabel('frequency')
plt.tight_layout()
plt.show()


print('Range final size:', '(', min(N_T), ',', max(N_T), ')')

#%%

#Rapport taille finale par rapport à l'initiale
rapport_tailleinitiale_finale = N_T/N0s
rapport_tailleinitiale_finale = pd.DataFrame(rapport_tailleinitiale_finale)

# Plot rapport

plt.figure()
rapport_tailleinitiale_finale.hist(bins = 100)
plt.title('Distribution du rapport taille finale sur l\'initiale')
plt.tight_layout()
plt.show()

#%%
sizes = pd.DataFrame(N[0,:,:])
test = sizes.T.stack()
test = pd.DataFrame(test)
test = test.reset_index(level=[0,1])
#test['farm_id'] = test.index.get_level_values(0)
#test['simul_time'] = test.index.get_level_values(1)
rapport_tailleinitiale_finale = N_T/N0s
test['rapport'] = pd.DataFrame(np.repeat(rapport_tailleinitiale_finale, nb_steps))
#test.drop(columns=['level_0', 'level_1'])

level_1 = np.array(test["level_1"])
a = level_1*delta
a = a.astype(int)
y = np.array(test[0])
rapport = np.array(test["rapport"])

#%%
fig, ax = plt.subplots(figsize=(8, 5)) #figsize=(15, 10)
scatter = ax.scatter(x=a, y=y, c=rapport, s=1, cmap="magma_r")
legend1 = ax.legend(scatter.get_label(), title="Rapport")
ax.add_artist(legend1)
ax.legend(fontsize=10)
plt.colorbar(scatter)
plt.xlabel('day')
plt.ylabel('herd size')
plt.show()
#%%



# Quels et combien d'animaux restent sans animaux:
# print(np.where(np.sum(sir_r[nb_steps-200], axis =1) == 0.))
print('Nb de fermes sans animaux à la fin:', len(np.where(np.sum(sir_r[0, nb_steps-200], axis =1) == 0.)[0]))

# Quels et combien d'animaux restent avec moins de 5 animaux
limit = 20.
# print(np.where(np.sum(sir_r[nb_steps-200], axis =1) < limit))
print('Nb de fermes avec moins de 5 animaux ) la fin:', len(np.where(np.sum(sir_r[0,nb_steps-200], axis =1) < limit)[0]))


#%%

pattern = [0,1,1,1,1,0]



#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
I_1 = action25_results[0, :, :, 2]/N_action25[0, :, :]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
    
    
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']

xpoints = decision_times_i

#days = []
#for i in range(nb_steps):
#    days.append(i*0.5)
    

plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'red')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')


plt.plot("Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

plt.title('Median proportion of SNV animals by herd on infected herds')
plt.xlabel("Day", fontweight = "bold")
plt.ylabel("Median proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)
plt.show()

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)

#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
a = action25_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action25_results[0, :, b, 2]/N_action25[0, :, b]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t, 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t, 90)
        I1_nonzeros[t, 2] = np.percentile(I_t, 10)
        I1_nonzeros[t, 3] = np.mean(I_t) 
        I1_nonzeros[t, 4] = np.percentile(I_t, 90)
        I1_nonzeros[t, 5] = np.percentile(I_t, 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']



xpoints = decision_times_i

days = []
for i in range(nb_steps):
    days.append(i*0.5)
    
plt.style.use("default")
iline = plt.plot(days, "Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'red')
mean = plt.plot(days, "Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot(days, "Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot(days, "Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

#plt.title('Prop. of I animals on infected herds with pattern ' + str(pattern))
plt.xlabel("day")
plt.ylabel("proportion of infected animals by herd")
legend = plt.legend(loc = 'upper right')
frame = legend.get_frame()



for p in xpoints*delta:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)


#%%

a = action25_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action25_results[0, :, b, 1]/action25_results[0, :, b, 0] # /N_action25[0, :, b]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 100)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 1)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 100)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 1)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'red')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

xpoints = decision_times_i


for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

plt.title('Prop. of SNV to I animals on infected herds with pattern ' + str(pattern))
plt.xlabel("Time", fontweight = "bold")
plt.ylabel("Proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#%%

a = action25_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action25_results[0, :, b, 1]# /N_action25[0, :, b]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 100)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 1)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 100)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 1)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros.cumsum(), linewidth = 2, color = 'red')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros.cumsum(), linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

xpoints = decision_times_i


for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros.cumsum(), linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros.cumsum(), linewidth = 2, color = 'lightgray')

plt.title('Prop. of SNV to I animals on infected herds with pattern ' + str(pattern))
plt.xlabel("Time", fontweight = "bold")
plt.ylabel("Proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)


#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
a = action25_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action25_results[0, :, b, 0]/N_action25[0, :, b]#/(action25_results[0, :, b, 0] + action25_results[0, :, b, 3])
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot(days, "Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'blue')
mean = plt.plot(days, "Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')

xpoints = decision_times_i*delta

days = []
for i in range(nb_steps):
    days.append(i*0.5)
    

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
#plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot(days,"Q90", "",  data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot(days,"Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

legend = plt.legend(loc = 'lower left')
frame = legend.get_frame()


for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')


#plt.title('Proportion of SNV animals by herd on infected herds ')
plt.xlabel("day")
plt.ylabel("proportion of animals by herd")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)


#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
a = action25_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action25_results[0, :, b, 3]/N_action25[0, :, b]# /(action25_results[0, :, b, 0] + action25_results[0, :, b, 3])
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'blue')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

plt.title('Proportion of SV animals by herd on infected herds ' + str(pattern))
plt.xlabel("Time", fontweight = "bold")
plt.ylabel("Median proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
a = action25_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action25_results[0, :, b, 7]/N_action25[0, :, b]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'green')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

plt.title('Proportion of R animals by herd on infected herds ' + str(pattern))
plt.xlabel("Time", fontweight = "bold")
plt.ylabel("Median proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#%%

pattern = [0,0,0,1,0,0]

#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
I_1 = action_results[0, :, :, 2]/N_action[0, :, :]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
    
    
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']

xpoints = decision_times_i*delta

days = []
for i in range(nb_steps):
    days.append(i*0.5)
    

plt.style.use("default")
iline = plt.plot(days, "Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'red')
mean = plt.plot(days,"Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')


plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot(days,"Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot(days,"Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

plt.title('Median proportion of SNV animals by herd on infected herds')
plt.xlabel("Day", fontweight = "bold")
plt.ylabel("Median proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)





#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
a = action_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action_results[0, :, b, 2]/N_action[0, :, b]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot(days, "Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'red')
mean = plt.plot(days, "Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')


#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot(days, "Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot(days,"Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
legend = plt.legend(loc = 'upper right')
frame = legend.get_frame()


xpoints = decision_times_i*delta
for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')


#plt.title('Prop. of I animals on infected herds with pattern ' + str(pattern))
plt.xlabel("day")
plt.ylabel("proportion of infected animals by herd")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)

#%%

I_1 = action_results[0, :, b, 1]/N_action[0, :, b]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'red')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

xpoints = decision_times_i
for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

plt.title('Prop. of SNV to I animals on infected herds with pattern ' + str(pattern))
plt.xlabel("Time", fontweight = "bold")
plt.ylabel("Proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#%%

a = action_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action_results[0, :, b, 1]/action_results[0, :, b, 0]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'red')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

xpoints = decision_times_i


for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

plt.title('Prop. of SNV to I animals on infected herds with pattern ' + str(pattern))
plt.xlabel("Time", fontweight = "bold")
plt.ylabel("Proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
I_1 = action_results[0, :, b, 0]/(N_action[0, :, b])
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'blue')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

xpoints = decision_times_i

for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

#plt.title('Proportion of SNV animals by herd on infected herds ' + str(pattern))
plt.xlabel("Time", fontweight = "bold")
plt.ylabel("Median proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
a = action_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)
I_1 = action_results[0, :, b, 0]/N_action[0, :, b]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot(days, "Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'blue')
mean = plt.plot(days,"Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')


#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot(days, "Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot(days, "Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

legend = plt.legend(loc = 'lower left')
frame = legend.get_frame()

for p in xpoints*delta:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')


# plt.title('Proportion of SV animals by herd on infected herds ' + str(pattern))
plt.xlabel("day")
plt.ylabel("proportion of SNV animals by herd")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)



#%%
# Prop Mediane du nb d'infectés sans compter les troupeaux qui n'ont pas d'infectés
I_1 = action_results[0, :, b, 7]/N_action[0, :, b]
I_1[np.isnan(I_1)] = 0
I1_nonzeros = np.zeros((nb_steps, 6))
for t in range(0,nb_steps):
    I_t = I_1[:, t]
    if I_t[I_t!= 0.0].size:
        I1_nonzeros[t, 0] = np.percentile(I_t[I_t!= 0.0], 50) 
        I1_nonzeros[t, 1] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 2] = np.percentile(I_t[I_t!= 0.0], 10)
        I1_nonzeros[t, 3] = np.mean(I_t[I_t!= 0.0]) 
        I1_nonzeros[t, 4] = np.percentile(I_t[I_t!= 0.0], 90)
        I1_nonzeros[t, 5] = np.percentile(I_t[I_t!= 0.0], 10)
    else:
        I1_nonzeros[t, 0] = 0
        I1_nonzeros[t, 1] = 0
        I1_nonzeros[t, 2] = 0
        I1_nonzeros[t, 3] = 0
        I1_nonzeros[t, 4] = 0
        I1_nonzeros[t, 5] = 0
        
I1_pd_nonzeros = pd.DataFrame(I1_nonzeros)
I1_pd_nonzeros.columns = ['Median', 'Q75', 'Q25', 'Mean', 'Q90', 'Q10']


plt.style.use("default")
iline = plt.plot("Median", "", data = I1_pd_nonzeros, linewidth = 2, color = 'green')
mean = plt.plot("Mean", "", data = I1_pd_nonzeros, linewidth = 2, color = 'tab:gray')
legend = plt.legend(loc = 5, bbox_to_anchor = (1.25,0.5))
frame = legend.get_frame()

for p in xpoints:
    plt.axvline(p, 0, 1, label='pyplot vertical line', linewidth = 0.5, color = 'gray')

#plt.plot(days,"Q75", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
#plt.plot(days,"Q25", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightpink')
plt.plot("Q90", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')
plt.plot("Q10", "", data = I1_pd_nonzeros, linewidth = 2, color = 'lightgray')

plt.title('Proportion of R animals by herd on infected herds ' + str(pattern))
plt.xlabel("Time", fontweight = "bold")
plt.ylabel("Median proportion of animals by herd", fontweight = "bold")
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_linewidth(0)

#sum_SIR = np.sum(sir_1 , axis = 1)
#mean = np.sum(sir_1, axis = 1)/L 
#median = np.percentile(sir_1, 50, axis = 1) 
#sd = np.std(sir, axis = 1)
#UC = np.percentile(sir, 75, axis = 1) # mean + (2.576* sd/np.sqrt(L)) # np.max(sir, axis = 1)# 
#LC = np.percentile(sir, 25, axis = 1) # mean + (2.576* sd/np.sqrt(L)) #np.min(sir, axis = 1)


#%%


import itertools
from itertools import product
all_combinations = pd.DataFrame(list(product([[0,1]]*len(decision_times_i))))


columns =  [[0,1]]*6
all_combinations = pd.DataFrame(list(itertools.product(*columns)))

all_combinations ['ColumnA'] = all_combinations[all_combinations .columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)


behavior_pd = pd.DataFrame(action_behaviors[0, decision_times_i,].T)
behavior_pd2 = behavior_pd.copy()


behavior_pd2['ColumnA'] = behavior_pd2[behavior_pd2.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)

behavior_pd3 = behavior_pd2.copy()

behavior_pd3 = behavior_pd3.groupby(['ColumnA']).size()
behavior_pd3 = behavior_pd3.reset_index(name='counts')
behavior_pd3['counts'] /= L 

behavior_pd3 = behavior_pd3.sort_values(by=['counts'], ascending=False)

all_combinations = all_combinations.merge(behavior_pd3,how='left', left_on='ColumnA', right_on='ColumnA')



actual = all_combinations.dropna().sort_values(by = 'counts', ascending=False)
#print(actual[actual['counts'] ]) #>= 0.01
a = actual[["ColumnA", "counts"]]# >= 0.01
#print(a['ColumnA'])

print(a)


print(a.to_latex())  

#%%
all_combinations = all_combinations.fillna(0.0)


all_combinations.plot.bar(x = 'ColumnA', y = 'counts',legend=False)

plt.xticks(rotation='vertical',fontsize=5)
plt.xlabel('pattern',fontsize=5)

#%%



behavior_pd = pd.DataFrame(action_behaviors[0, decision_times_i,].T)
behavior_pd = behavior_pd.sort_values(by=[0,1,2,3,4,5])


behavior_pd2 = behavior_pd.copy()
behavior_pd3 = behavior_pd.copy()
behavior_pd3['ColumnA'] = behavior_pd3[behavior_pd3.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)


plt.hist(np.array(behavior_pd3['ColumnA']), bins = int(2**6),
         weights=np.zeros_like(behavior_pd3['ColumnA']) + 1. /behavior_pd3['ColumnA'].size)
plt.xticks(rotation='vertical',fontsize=5)

#%%


behavior_pd3 = behavior_pd3.groupby(['ColumnA']).size()
behavior_pd3 = behavior_pd3.reset_index(name='counts')
behavior_pd3['counts'] /= L 

behavior_pd3 = behavior_pd3.sort_values(by=['counts'])

bins = np.arange(0,len(behavior_pd3))

behavior_pd2['mean'] = behavior_pd2.mean(axis = 1)
plt.hist(np.array(behavior_pd2['mean']), bins = 64, weights=np.zeros_like(behavior_pd2['mean']) + 1. /behavior_pd2['mean'].size)

behavior_pd2['ColumnA'] = behavior_pd2[behavior_pd2.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)


actual_behaviors = behavior_pd2.drop_duplicates()


#%%



import itertools
from itertools import product
all_combinations = pd.DataFrame(list(product([[0,1]]*len(decision_times_i))))


columns =  [[0,1]]*6
all_combinations = pd.DataFrame(list(itertools.product(*columns)))

all_combinations ['ColumnA'] = all_combinations[all_combinations .columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)


behavior_pd = pd.DataFrame(action25_behaviors[0, decision_times_i,].T)
behavior_pd2 = behavior_pd.copy()


behavior_pd2['ColumnA'] = behavior_pd2[behavior_pd2.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)

behavior_pd3 = behavior_pd2.copy()

behavior_pd3 = behavior_pd3.groupby(['ColumnA']).size()
behavior_pd3 = behavior_pd3.reset_index(name='counts')
behavior_pd3['counts'] /= L 

behavior_pd3 = behavior_pd3.sort_values(by=['counts'])

all_combinations = all_combinations.merge(behavior_pd3,how='left', left_on='ColumnA', right_on='ColumnA')

#actual = all_combinations.dropna().sort_values(by = 'counts', ascending=False)
#print(actual[actual['counts'] >= 0.01])


actual = all_combinations.dropna().sort_values(by = 'counts', ascending=False)
#print(actual[actual['counts'] ]) #>= 0.01
a = actual[["ColumnA", "counts"]]# >= 0.01
#print(a['ColumnA'])

print(a)


print(a.to_latex(index=False))  

#%%
all_combinations = all_combinations.fillna(0.0)


all_combinations.plot.bar(x = 'ColumnA', y = 'counts',legend=False)

plt.xticks(rotation='vertical',fontsize=5)
plt.xlabel('pattern',fontsize=5)

#%%



behavior_pd = pd.DataFrame(action25_behaviors[0, decision_times_i,].T)
behavior_pd = behavior_pd.sort_values(by=[0,1,2,3,4,5])


behavior_pd2 = behavior_pd.copy()
behavior_pd3 = behavior_pd.copy()
behavior_pd3['ColumnA'] = behavior_pd3[behavior_pd3.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)


plt.hist(np.array(behavior_pd3['ColumnA']), bins = int(2**6),
         weights=np.zeros_like(behavior_pd3['ColumnA']) + 1. /behavior_pd3['ColumnA'].size)
plt.xticks(rotation='vertical',fontsize=5)

#%%


behavior_pd3 = behavior_pd3.groupby(['ColumnA']).size()
behavior_pd3 = behavior_pd3.reset_index(name='counts')
behavior_pd3['counts'] /= L 

behavior_pd3 = behavior_pd3.sort_values(by=['counts'])

bins = np.arange(0,len(behavior_pd3))

behavior_pd2['mean'] = behavior_pd2.mean(axis = 1)
plt.hist(np.array(behavior_pd2['mean']), bins = 64, weights=np.zeros_like(behavior_pd2['mean']) + 1. /behavior_pd2['mean'].size)

behavior_pd2['ColumnA'] = behavior_pd2[behavior_pd2.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)


actual_behaviors = behavior_pd2.drop_duplicates()

#%%



behavior_pd = pd.DataFrame(pd.DataFrame(action25_behaviors[0, decision_times_i,].T))
behavior_pd4 = behavior_pd.copy()
behavior_pd4 = behavior_pd4.replace(0, 'NV')
behavior_pd4 = behavior_pd4.replace(1, 'V')

#%%

for col in range(behavior_pd4.shape[1]):
    behavior_pd4[col] = behavior_pd4[col] + str(col)

#%%
behavior_pd4['id'] = behavior_pd4.index
counts_behavior = behavior_pd4.groupby(list(range(len(decision_times_i))))["id"].count().reset_index(name="count")
counts_behavior['target'] = counts_behavior.index
counts_behavior.rename(columns={0:'source',
                                'count':'value',
                                1:'category1',
                                2:'category2',
                                3:'category3',
                                4:'category4',
                                5:'category5'},
                       
                       inplace=True)


#%%
import floweaver
#from floweaver import *
# set the "nodes" - aka grouping spots. (Node names here aren't important)
nodes = {
    'start': floweaver.ProcessGroup(list(counts_behavior['source'])),
    'category1': floweaver.Waypoint(floweaver.Partition.Simple('category1', counts_behavior['category1'].unique())),
    'category2': floweaver.Waypoint(floweaver.Partition.Simple('category2', counts_behavior['category2'].unique())),
    'category3': floweaver.Waypoint(floweaver.Partition.Simple('category3', counts_behavior['category3'].unique())),
    'category4': floweaver.Waypoint(floweaver.Partition.Simple('category4', counts_behavior['category4'].unique())),
    'category5': floweaver.Waypoint(floweaver.Partition.Simple('category5', counts_behavior['category5'].unique())),
    'end': floweaver.ProcessGroup(list(counts_behavior['target']))
}

# set the order of the nodes left to right
ordering = [['start'], 
            ['category1'],
            ['category2'],
            ['category3'],
            ['category4'],
            ['end']]

# set the "bundle" of connections you want to show
bundles = [floweaver.Bundle('start', 'end', waypoints=['category1', 'category2',
                                             'category3', 'category4'])]

# add the partitions
partner = floweaver.Partition.Simple('source', counts_behavior['source'].unique())
last = floweaver.Partition.Simple('target', counts_behavior['target'].unique())
cat = floweaver.Partition.Simple('category5', counts_behavior['category5'].unique())

nodes['start'].partition = partner
nodes['end'].partition = floweaver.Partition.Simple('category5', counts_behavior['category5'].unique())

# add the color palette
#palette = {'NV_0': 'red', 'V_0': 'blue'}
# 'Set3_12'

# create the sankey diagram
sdd = floweaver.SankeyDefinition(nodes, bundles, ordering, flow_partition=last)

# display the sankey diagram
floweaver.weave(sdd, counts_behavior, palette = 'Paired_12', measures='value').to_widget()#.auto_save_png('Sankey_pretty_1.png')

#%%

#Indices des parents de chaque troupeau

parents_edges_compact = []
for k in range(0,L):
    parents_k = []
    for w in theta_edges[theta_edges[:,1]== k]:
        parents_k.append(int(w[0]))
    parents_edges_compact.append([k, parents_k ])
parents_edges_compact = np.array(parents_edges_compact,dtype=object)

#%%
#Indices des troupeaux dans chaque pattern

behavior_pd = pd.DataFrame(action25_behaviors[0, decision_times_i,].T)
#behavior_pd = behavior_pd.sort_values(by=[0,1,2,3,4,5])
behavior_pd2 = behavior_pd.copy()
behavior_pd2['ColumnA'] = behavior_pd[behavior_pd.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)
pattern_by_farm = behavior_pd2['ColumnA']


#Matrice: pour chaque ferme: nb de parents avec chaque pattern possible

all_patterns = list(all_combinations['ColumnA'])
parents_edges_patterns = np.zeros((L, len(all_patterns)))
for k in range(0,L):
    parents_k =  parents_edges_compact[k,1]
    behavior_parents = []
    for l in range(len(parents_k)):
        behavior_parents.append(pattern_by_farm[parents_k[l]])
    parents_edges_patterns_k = list(map(behavior_parents.count, all_patterns))
    parents_edges_patterns[k] = np.array(parents_edges_patterns_k)/len(parents_k)
        
parents_edges_patterns_pd = pd.DataFrame(parents_edges_patterns) 
parents_edges_patterns_pd['ColumnA'] =  behavior_pd2['ColumnA']


x = parents_edges_patterns_pd.groupby(by = 'ColumnA').mean()
x.columns = list(all_combinations['ColumnA'])
x =x*100
#x = x.T
#pattern_seller = x.index
#x.insert(0, "pattern_seller", pattern_seller)


a = x.iloc[7].sort_values(ascending=False)

print(a.to_latex(float_format="%.2f"))  

#%%


#In theta edges de chaque troupeau

in_theta_edges_compact = []
for k in range(0,L):
    theta_parents_k = []
    for w in theta_edges[theta_edges[:,1]== k]:
        theta_parents_k.append(w[2])
    theta_parents_k= theta_parents_k/sum(theta_parents_k)    
    in_theta_edges_compact.append([k,theta_parents_k])
in_theta_edges_compact = np.array(in_theta_edges_compact,dtype=object)

#%%

#Matrice: pour chaque ferme: in_theta_j avec chaque pattern possible

all_patterns = list(all_combinations['ColumnA'])
thetas_parents_patterns = np.zeros((L, len(all_patterns)))
for k in range(0,L):
    parents_k =  parents_edges_compact[k,1]
    in_theta_k = in_theta_edges_compact[k,1]
    behavior_parents = []
    for l in range(len(parents_k)):
        behavior_parents.append(pattern_by_farm[parents_k[l]])
    pattern_theta_k = pd.DataFrame((behavior_parents, in_theta_k)).T
    pattern_theta_k[1] = pd.to_numeric(pattern_theta_k[1])
    pattern_theta_k = pattern_theta_k.groupby(by = 0).mean()
    all_patterns_df = pd.DataFrame(all_patterns)
    thetas_k_pattern = pd.merge(all_patterns_df, pattern_theta_k, on = 0, how = 'outer').fillna(0)
    thetas_parents_patterns[k] = thetas_k_pattern[1]
#%%     
thetas_parents_patterns_pd = pd.DataFrame(thetas_parents_patterns).astype(np.float16)
thetas_parents_patterns_pd['ColumnA'] = behavior_pd2['ColumnA']
y = thetas_parents_patterns_pd.groupby(by = 'ColumnA').mean()
y.columns = list(all_combinations['ColumnA'])
y = y.div(y.sum(axis=1), axis=0)*100

#%%
sum(x.loc['0,0,0,1,0,1'])

#%%


a = action25_behaviors[0, decision_times_i, :].T
b = np.all(a == pattern, axis = 1)


#Plot ventes des ventes cumulées  a travers le temps
achats_by_farm = pd.DataFrame(achats_action25[0, :, b].T).cumsum()
plt.figure()
achats_by_farm.plot(legend=False)
plt.tight_layout()
plt.show()


#Plot ventes des ventes cumulées  a travers le temps
ventes_by_farm = pd.DataFrame(ventes_action25[0, :, b].T).cumsum()
plt.figure()
ventes_by_farm.plot(legend=False)
plt.tight_layout()
plt.show()

#%%

#Indices des troupeaux dans chaque pattern

behavior_pd = pd.DataFrame(action25_behaviors[0, decision_times_i,].T)
#behavior_pd = behavior_pd.sort_values(by=[0,1,2,3,4,5])
behavior_pd2 = behavior_pd.copy()
behavior_pd2['ColumnA'] = behavior_pd[behavior_pd.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)
pattern_by_farm = behavior_pd2['ColumnA']

pattern_by_farm


#Test:
time_seuil = int(nb_steps)
time_seuil_in_days = int(time_seuil*delta)
pattern_pd = pd.DataFrame(pattern_by_farm)

achats_time = np.sum(achats_action25[0, :time_seuil], axis = 0)
sells_time = np.sum(ventes_action25[0, :time_seuil], axis = 0)

which = pattern_pd[pattern_pd['ColumnA'] == '0,1,1,1,1,1'].index
df_2= df_out_degrees.copy()
df_2['in-strength'] = (achats_time) 
df_2['out-strength'] =  (sells_time) 
df_2 = df_2.iloc[which, :]
df_2['pattern'] = '0,1,1,1,1,1'

which = pattern_pd[pattern_pd['ColumnA'] == '0,0,0,0,0,0'].index
df_out_degrees = df_out_degrees.rename(columns={0: "out-degree"})
df_1 = df_out_degrees.copy()
df_1 ['in-strength'] = (achats_time) 
df_1 ['out-strength'] =  (sells_time) 
df_1  = df_1.iloc[which, :]
df_1 ['pattern'] = '0,0,0,0,0,0'

which = pattern_pd[pattern_pd['ColumnA'] == '0,0,1,1,1,1'].index
df_3= df_out_degrees.copy()
df_3['in-strength'] =  (achats_time) 
df_3['out-strength'] =  (sells_time) 
df_3 = df_3.iloc[which, :]
df_3['pattern'] = '0,0,1,1,1,1'

which = pattern_pd[pattern_pd['ColumnA'] == '0,1,0,0,0,0'].index
df_4 = df_out_degrees.copy()
df_4['in-strength'] =  (achats_time) 
df_4['out-strength'] =  (sells_time) 
df_4 = df_4.iloc[which, :]
df_4['pattern'] = '0,1,0,0,0,0'

df_accordingq = pd.concat([df_2,df_1, df_3, df_4])

#%%
#f, ax = plt.subplots(figsize=(7, 7))
#ax.set(, yscale="log")
g = sns.pairplot(df_accordingq, vars = ['N0s', 'out-degree', 'in-strength'],
                 hue="pattern", palette=['#0571b0', '#ca0020', '#92c5de', '#f4a582'])

# Set the `yscale`
g.set(xscale="log", yscale="log")

a = "Exploration des troupeaux par pattern des vaccination pour les patterns les plus fréquents"
#g.fig.suptitle(a, y = 1.01)



#%%

#Indices des troupeaux dans chaque pattern

behavior_pd = pd.DataFrame(action25_behaviors[0, decision_times_i,].T)
#behavior_pd = behavior_pd.sort_values(by=[0,1,2,3,4,5])
behavior_pd2 = behavior_pd.copy()
behavior_pd2['ColumnA'] = behavior_pd[behavior_pd.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)
pattern_by_farm = behavior_pd2['ColumnA']
behavior_pd3 = behavior_pd2.copy()

#Test:
time_seuil = int(nb_steps)
time_seuil_in_days = int(time_seuil*delta)
pattern_pd = pd.DataFrame(pattern_by_farm)

achats_time = np.sum(achats_action25[0, :time_seuil], axis = 0)
sells_time = np.sum(ventes_action25[0, :time_seuil], axis = 0)
df_1 = df_out_degrees.copy()
df_1 ['buys'] = achats_time
df_1 ['sells'] =  sells_time
df_1['pattern'] = pattern_by_farm 
df_1 = df_1.groupby('pattern').mean()



behavior_pd3 = behavior_pd3.groupby(['ColumnA']).size()
behavior_pd3 = behavior_pd3.reset_index(name='counts')
behavior_pd3['counts'] /= L 

df_1['counts'] = np.array(behavior_pd3['counts'])
df_1['patterns'] = df_1.index



#%%

#Indices des troupeaux dans chaque pattern

behavior_pd = pd.DataFrame(action_behaviors[0, decision_times_i,].T)
#behavior_pd = behavior_pd.sort_values(by=[0,1,2,3,4,5])
behavior_pd2 = behavior_pd.copy()
behavior_pd2['ColumnA'] = behavior_pd[behavior_pd.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)
pattern_by_farm = behavior_pd2['ColumnA']
behavior_pd3 = behavior_pd2.copy()

#Test:
time_seuil = int(nb_steps)
time_seuil_in_days = int(time_seuil*delta)
pattern_pd = pd.DataFrame(pattern_by_farm)

achats_time = np.sum(achats_action[0, :time_seuil], axis = 0)
sells_time = np.sum(ventes_action[0, :time_seuil], axis = 0)
df_1 = df_out_degrees.copy()
df_1.rename(columns={0: "out-deg"})
df_1 ['buys'] = achats_time
df_1 ['sells'] =  sells_time
df_1['pattern'] = pattern_by_farm 
df_1 = df_1.groupby('pattern').mean()



behavior_pd3 = behavior_pd3.groupby(['ColumnA']).size()
behavior_pd3 = behavior_pd3.reset_index(name='counts')
behavior_pd3['counts'] /= L 

df_1['counts'] = np.array(behavior_pd3['counts'])
df_1['patterns'] = df_1.index

#%%


#Indices des troupeaux dans chaque pattern

behavior_pd = pd.DataFrame(action_behaviors[0, decision_times_i,].T)
#behavior_pd = behavior_pd.sort_values(by=[0,1,2,3,4,5])
behavior_pd2 = behavior_pd.copy()
behavior_pd2['ColumnA'] = behavior_pd[behavior_pd.columns[:]].apply(
    lambda x: ','.join(x.astype(int).dropna().astype(str)),
    axis=1
)
pattern_by_farm = behavior_pd2['ColumnA']

pattern_by_farm


#Test:
time_seuil = int(nb_steps)
time_seuil_in_days = int(time_seuil*delta)
pattern_pd = pd.DataFrame(pattern_by_farm)

achats_time = np.sum(achats_action[0, :time_seuil], axis = 0)
sells_time = np.sum(ventes_action[0, :time_seuil], axis = 0)

which = pattern_pd[pattern_pd['ColumnA'] == '0,0,0,0,0,0'].index
df_out_degrees = df_out_degrees.rename(columns={0: "out-degree"})
df_1 = df_out_degrees.copy()
df_1 ['in-strength'] = (achats_time) 
df_1 ['out-strength'] =  (sells_time) 
df_1  = df_1.iloc[which, :]
df_1 ['pattern'] = '0,0,0,0,0,0'


which = pattern_pd[pattern_pd['ColumnA'] == '0,0,0,0,0,1'].index
df_2= df_out_degrees.copy()
df_2['in-strength'] = (achats_time) 
df_2['out-strength'] =  (sells_time) 
df_2 = df_2.iloc[which, :]
df_2['pattern'] = '0,0,0,0,0,1'

which = pattern_pd[pattern_pd['ColumnA'] == '0,0,1,1,1,1'].index
df_3= df_out_degrees.copy()
df_3['in-strength'] =  (achats_time) 
df_3['out-strength'] =  (sells_time) 
df_3 = df_3.iloc[which, :]
df_3['pattern'] = '0,0,1,1,1,1'

which = pattern_pd[pattern_pd['ColumnA'] == '0,1,0,0,0,0'].index
df_4 = df_out_degrees.copy()
df_4['in-strength'] =  (achats_time) 
df_4['out-strength'] =  (sells_time) 
df_4 = df_4.iloc[which, :]
df_4['pattern'] = '0,1,0,0,0,0'

df_accordingq = pd.concat([df_2,df_1, df_3, df_4])

#%%
#f, ax = plt.subplots(figsize=(7, 7))
#ax.set(, yscale="log")
g = sns.pairplot(df_accordingq, vars = ['N0s', 'out-degree', 'in-strength'],
                 hue="pattern", palette=['#0571b0', '#ca0020', '#92c5de', '#f4a582'])

# Set the `yscale`
g.set(xscale="log", yscale="log")

a = "Exploration des troupeaux par pattern des vaccination pour les patterns les plus fréquents"
#g.fig.suptitle(a, y = 1.01)




