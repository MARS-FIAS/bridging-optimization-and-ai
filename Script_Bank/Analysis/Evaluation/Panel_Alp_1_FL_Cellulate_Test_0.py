# Reference listing: scientific execution uses the cavity_sbi package.
# Filesystem markers below replace the original machine-specific prefixes.
raise RuntimeError(
    "Reference listing. Use python -m cavity_sbi --help for the supported workflow."
)

###########################
######## Panel Alp ########
###########################

#%%# Catalyzer

import os
import sys
path = 'REFERENCE_PANEL_RESOURCES'
sys.path.append(path)
# path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(path)
import numpy as np
import torch
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt

fig_resolution = 500 # {250, 375, 500, 750, 1000}
plt.rcParams['figure.dpi'] = fig_resolution
plt.rcParams['savefig.dpi'] = fig_resolution
fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}

#%%# Retrieve Posterior Data

path_service = 'REFERENCE_ANALYSIS_EVALUATION/'
sys.path.append(path_service)

from Optimize_Exploration_Analysis_Service_0 import retrieve_posteriors_mapes

_acts = 5
data_paths = ['Panel_Alp_1_FL_Cellulate']*_acts
acts = list(range(0, _acts))
observers = [1]*_acts

cellulates = [(_, _) for _ in range(1, _acts+1)] + [(5, 3, 1), (7, 5, 3, 1), (9, 7, 5, 3, 1)]

posterior_data = {cellulate: {'posteriors': None, 'mapes': None} for cellulate in cellulates}

for cellulate in cellulates:
    curbs = ['_'.join(map(str, cellulate))]*_acts
    posteriors, mapes = retrieve_posteriors_mapes(data_paths, acts, observers, curbs)
    posterior_data[cellulate]['posteriors'] = posteriors
    posterior_data[cellulate]['mapes'] = mapes

#%%# Compare Parameter Value (MAPE!) Sets

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!
almas = [0.5, 0.75]
lisps = ['', '-']
lites = [2, 3]
marks = ['', '.']
cocos = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'] + ['tab:pink', 'tab:olive', 'tab:cyan']

#%%# Extract Posterior Samples (Pick!)

from Optimize_Exploration_Analysis_Service_0 import synthesizer_post

pick = ((9, 7, 5, 3, 1), 4) # ('Cellulate', 'Act')
pick_posterior = posterior_data[pick[0]]['posteriors'][pick[1]]
_pick_posterior_samples, _pick_theta = synthesizer_post(posterior = pick_posterior, observation = None, posterior_sample_shape = tuple([250000]))

pick_posterior_samples = _pick_posterior_samples.numpy()
pick_theta = _pick_theta['mape'].numpy()
print(f"Synthesize Post!\n{' '*4}Pick Cellulate! {pick[0]}\n{' '*4}Pick Act! {pick[1]}\n{' '*4}Posterior Sample Mat Shape {pick_posterior_samples.shape}\n{' '*4}MAPE Vet Shape {pick_theta.shape}")

#%%# Extract MAPE Sets

_pick_mapes = {cellulate: posterior_data[cellulate]['mapes'][pick[1]] for cellulate in cellulates}

#%%# Compare Parameter Value (MAPE!) Sets [Separate]

para_set_true = pick_posterior.parameter_set_true
paras = list(para_set_true.keys())
nork_paras = [para_set_true[paras[_]][1][1]-para_set_true[paras[_]][1][0] for _ in range(0, len(paras))]

pick_mapes = {cellulate: None for cellulate in cellulates}
for cellulate in cellulates:
    mape = _pick_mapes[cellulate]
    pick_mapes[cellulate] = np.array([mape[_]/nork_paras[_] for _ in range(0, len(paras))])

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig0'
fig_forts = ['tiff', 'svg', 'eps', 'pdf', 'png'] # {TIFF, SVG, EPS, PDF, PNG}
fig_keros = [{'compression': 'tiff_lzw'}, None, None, None, None]
fig_flags = [fig_kero is not None for fig_kero in fig_keros]

for fig_fort in fig_forts:
    fig_fort_index = fig_forts.index(fig_fort)
    fig_kero = fig_keros[fig_fort_index]
    fig_flag = fig_flags[fig_fort_index]
    fig_clue = f'{fig_path}/{fig_nick}.{fig_fort}'
    print(fig_clue)
    if fig_flag:
        fig.savefig(fname = fig_clue, dpi = fig_resolution, format = fig_fort, pil_kwargs = fig_kero)
    else:
        fig.savefig(fname = fig_clue, dpi = fig_resolution, format = fig_fort)
