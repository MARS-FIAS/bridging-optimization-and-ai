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

path_service = 'REFERENCE_ANALYSIS/'
sys.path.append(path_service)

from Fig0_Services import retrieve_posteriors_mapes

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
cocos = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple'] + ['tab:cyan', 'tab:olive', 'tab:pink']

#%%# Extract Posterior Samples (Pick!)

from Fig0_Services import synthesizer_post

pick = ((5, 3, 1), 4) # ('Cellulate', 'Act')
pick_posterior = posterior_data[pick[0]]['posteriors'][pick[1]]
pick_posterior_samples_temp, pick_theta_temp = synthesizer_post(posterior = pick_posterior, observation = None, posterior_sample_shape = tuple([250000]))

_pick_posterior_samples = pick_posterior_samples_temp.numpy()
_pick_theta = pick_theta_temp['mape'].numpy()
print(f"Synthesize Post!\n{' '*4}Pick Cellulate! {pick[0]}\n{' '*4}Pick Act! {pick[1]}\n{' '*4}Posterior Sample Mat Shape {_pick_posterior_samples.shape}\n{' '*4}MAPE Vet Shape {_pick_theta.shape}")

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

pick_theta = np.array([int(np.round(_, 0)) if _ > 1 else np.round(_, 2) for _ in _pick_theta])
pick_posterior_samples = np.copy(_pick_posterior_samples)
for para in paras:
    para_index = paras.index(para)
    pick_theta[para_index] /= nork_paras[para_index]
    pick_posterior_samples[:, para_index] /= nork_paras[para_index]

#%%# Compare Parameter Value (MAPE!) Sets [Separate] Distance!

from scipy.spatial import distance

m = 1 # {0, 1, 2}
metrics = [None, 'cityblock', 'euclidean']
_metrics = [None, 'L1', 'L2']
metric = metrics[m]
_metric = _metrics[m]

X = np.array(list(pick_mapes.values()))
_Y = distance.squareform(distance.pdist(X, metric))
Y = _Y/np.power(X.shape[1], 1/m)
Z = np.round(100*Y, 2)
print(Z)

def make_heat_mask(heat_mate):
    heat_mask = np.invert(heat_mate.astype(bool))
    rows, cols = heat_mate.shape
    for row in range(rows):
        for col in range(cols):
            if col < row:
                heat_mask[row, col] = True
            elif col == row:
                heat_mask[row, col] = False
    return heat_mask

#%%# Compare Parameter Value (MAPE!) Sets [Separate] Distance Plot!

from seaborn import heatmap, light_palette

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!

prey = 100
n_colors = int(prey/2.5) # {1, ..., 100}
colors = ['darkcyan', 'royalblue', 'steelblue', 'midnightblue']
cmap = light_palette(color = colors[3], n_colors = n_colors, reverse = 0, as_cmap = False)

tick_step = (prey-0)/n_colors
ticks = np.arange(start = 0, stop = prey+tick_step, step = tick_step).tolist()
_tick_labels = [tick if ticks.index(tick) % 4 == 0 else None for tick in ticks]
tick_labels = [int(tick_label) if tick_label is not None and np.round(tick_label) == tick_label else tick_label for tick_label in _tick_labels]

heat_tick_step = 1
heat_ticks = np.arange(start = 0.5, stop = len(cellulates), step = heat_tick_step).tolist()
sizes = [sum(cellulate) for cellulate in cellulates]

fig_rows = 1
fig_cols = 1
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = 'constrained')
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)

heat_mask = make_heat_mask(Z)
heat_map = heatmap(data = Z, mask = heat_mask, cmap = cmap, vmin = 0, vmax = prey, linewidths = 0, linecolor = 'white', annot = True, fmt = '.4g', annot_kws = {'fontsize': font_size_bet}, cbar = True, square = True, ax = axe_mat[0, 0], cbar_kws = {'ticks': None, 'location': 'bottom', 'fraction': 0.05, 'pad': 0.025})
heat_map.set_xticks(ticks = heat_ticks, labels = cellulates, fontsize = font_size_bet, rotation = 15)
heat_map.set_yticks(ticks = heat_ticks, labels = sizes, fontsize = font_size_bet, rotation = 15)
axe_mat[0, 0].vlines(x = range(0, len(cellulates)+1), ymin = 0, ymax = len(cellulates)+1, colors = 'w', linewidths = 2, alpha = 1)
axe_mat[0, 0].hlines(y = range(0, len(cellulates)+1), xmin = 0, xmax = len(cellulates)+1, colors = 'w', linewidths = 2, alpha = 1)
axe_mat[0, 0].figure.axes[-1].set_xticks(ticks = ticks, labels = tick_labels, fontsize = font_size_bet)
axe_mat[0, 0].set_title(label = f'Parameter Value Set (Comparison)\n{X.shape[1]}-Dimensional Vector Space · {_metric} Distance [%]', x = 0.5, y = 1, pad = 7, fontsize = font_size_alp)
# axe_mat[0, 0].set_title(label = '[A]', loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold', position = (-0.0725, 1))
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig4'
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
