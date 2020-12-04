#!/usr/bin/env python
# coding: utf-8

'''
Copyright or © or Copr. [INRAE]

Contributor(s) : [Lina Cristancho Fajardo]  ([2020])

[lina.cristancho-fajardo@inrae.fr]

This software is a computer program whose purpose is to simulate the stochastic spread of a pathogen on an animal trade-network,
by generating farmer's control (vaccination) decisions on the pathogen's spread according to a certain decision mechanism. 

This software is governed by the [CeCILL-B] license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the [CeCILL-B]
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

*The fact that you are presently reading this means that you have had
knowledge of the [CeCILL-B] license and that you accept its terms.*
'''

# Import modules
import numpy as np
import random
import numba

def proba_edges(L, theta_edges, delta):
    '''
    Function that converts theta_ij into probabilities
    Uses theta_edges, delta (fixed parameters)
    '''
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


#### Epidemic functions definition ####

def create_initial_state(k, N0s, prop_infected_in_node):
    '''Function for creating the initial state of a herd with the given initial size and prop of infected animals'''
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


@numba.jit(nopython=True)
def vec_multinomial(L, prob_matrix, sizes, res):
    '''
    Optimized function for drawing L multinomial samples, each with different probabilities
    for SIRstep_vectorized() function
    '''
    for i in range(L):
        res[i] = np.random.multinomial(n = sizes[i], pvals = prob_matrix[i])
    return res 


def SIRstep_vectorized(L, current_states, capacities, demo_params, epid_params, fixed_epid_probas, thetas, 
                       eff_reduceinfec, eff_protect, simul, delta):
    '''Fonction of a SIR step given the current states of all herds: '''
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
    conditioned_births = (capacities - N > 0)*min(births, capacities - N) # Actual births are limited by herd capacity 
    Snv = Snv_sample[:,3] + conditioned_births
    Sv = Sv_sample[:,3]
    Inv = Inv_sample[:,3] + d_SnvI
    Iv = Iv_sample[:,3] + d_SvI
    R = R_sample[:,2] + d_InvR + d_IvR
    Snv_out, Inv_out, Sv_out, Iv_out = Snv_sample[:,2], Inv_sample[:,2], Sv_sample[:,2], Iv_sample[:,2]
    R_out = R_sample[:,1]
    
    # Return list of two arrays: current state, and exports by compartment.
    return np.array([Snv, d_SnvI, Inv, Sv, d_SvI, Iv, d_InvR + d_IvR, R]).T.astype(int), np.array([Snv_out, Inv_out, Sv_out, Iv_out, R_out]).T.astype(int)


@numba.jit(nopython=True)
def vec_exports_i(out_k, p_out_k):
    '''Optimized step fonction for assigning exports '''
    nb_neighb = len(p_out_k)
    res_k = np.zeros((5,nb_neighb))
    for i in range(5):
        res_k[i] = np.random.multinomial(out_k[i], p_out_k)
    return res_k.T


def vec_exports(L, thetas, probs_exports, outs):
    '''Optimized full fonction for assigning exports '''
    res = []
    for k in range(L):
        theta_k, p_out_k = thetas[k], probs_exports[k]
        if theta_k != 0:
            res_k = vec_exports_i(outs[k], p_out_k)
            res.append(res_k)
    return res

# Functions for defining the decision mechanism

def nothing(simul, L, *args):
    '''Farmers never vaccinate'''
    return np.zeros(L)

def always(simul, L, *args):
     '''All Farmers vaccinate'''
     return np.ones(L)

def draw(weights):
    '''Fonction for drawing an option according to weights'''
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

def log_exp_Sum(log_weights):
    '''Log of the sum of the exponential Log weights'''
    exp_logw = []
    for log_w in log_weights:
        exp_logw.append(np.exp(log_w))
    return np.log(sum(exp_logw))

def expw(simul, L, mean_rewards, counts, vaccinators, non_vaccinators, relat_reward, decision_times,
         log_weights, kappas, *args):
    '''Expw mechanism'''
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
    
    '''Neigh_expw mechanism'''
    
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


@numba.jit(nopython=True)
def vaccinate(current_states, vacc):
    '''Optimized function for applying the decision: SNV pass to SV, or SV pass to SNV'''
    vaccinators = np.where(vacc == 1)[0]
    non_vaccinators = np.where(vacc != 1)[0]
    
    states_vaccinators = np.copy(current_states[vaccinators,:])
    current_states[vaccinators,3] = states_vaccinators[:,0] + states_vaccinators[:,3]
    current_states[vaccinators,0] = np.zeros(len(vaccinators))
    
    states_non_vaccinators = np.copy(current_states[non_vaccinators,:])
    current_states[non_vaccinators,0] = states_non_vaccinators[:,0] + states_non_vaccinators[:,3]
    current_states[non_vaccinators,3] = np.zeros(len(non_vaccinators))
        
    return current_states


def path(initial_states, demo_params, epid_params, eco_params, fixed_epid_probas,
         neighbors_list, parents_list, probs_exports, duration_decision, eff_reduceinfec, eff_protect,
         thetas, delta, nb_steps, nexpw_params, theta_edges_compact,
         mechanism = 'neighb_expw'):
    
    '''Fonction for epidemic-decision path for all herds'''
    
    #Initialization
    L = len(initial_states)
    all_states = np.zeros((nb_steps, L, 8), dtype=int) 
    all_states[0] = np.copy(initial_states) #Snv, SnvI, Inv , Sv, SvI, Iv , IR, R
    ventes_byfarm = np.zeros((nb_steps, L))
    achats_byfarm = np.zeros((nb_steps, L))
    capacities = np.sum(initial_states, axis = 1)*1.5
    
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
        w_nv = 1.
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
                relat_reward =  reward_dec/Nt_sum
                vaccinators = np.where(prev_decision == 1.)
                non_vaccinators = np.where(prev_decision == 0.)
                
            #Take decision
            decisions[simul] = decision_function(simul, L, mean_rewards, counts, vaccinators, non_vaccinators, 
                                                 relat_reward, decision_times, log_weights, kappas, rhos, prev_decision,
                                                 theta_edges_compact)

            #Record decisions
            decisions_simul = decisions[simul]
            
            #Decisions are applied here 
            current_states = vaccinate(current_states, decisions_simul)
        
        ###################################################################
        #Change states
        prev_N = np.sum(current_states, axis = 1)
        current_states, outs = SIRstep_vectorized(L, current_states, capacities,
                                                  demo_params, epid_params, fixed_epid_probas, thetas,
                                                  eff_reduceinfec, eff_protect, simul, delta)
        ventes_byfarm[simul] = np.sum(outs, axis = 1)
        ###################################################################  
        #Assign exports
        exports = np.concatenate(vec_exports(L, thetas, probs_exports, outs))
        ####################################################################
    
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

