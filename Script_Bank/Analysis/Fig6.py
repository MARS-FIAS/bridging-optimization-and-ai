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

pick = ((5, 3, 1), 4) # ('Cellulate', 'Act') # pick = ((3, 3), 4) # pick = ((9, 7, 5, 3, 1), 4)
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

#%%# Make Heat Core Codas!

from Fig0_Services import synthesizer_post_coda

font_size_tick = 17
font_size_label = 19
plt.rcParams['xtick.labelsize'] = font_size_tick
plt.rcParams['axes.labelsize'] = font_size_label
plt.rcParams['axes.labelweight'] = 'normal'

para_key_label_set = [
    r'$\mathit{Nanog}$_NANOG', r'$\mathit{Gata6}$_GATA6', r'$\mathit{Fgf4}$_NANOG', r'$\mathit{Gata6}$_A-ERK', r'$\mathit{Gata6}$_NANOG', r'$\mathit{Nanog}$_GATA6', r'$\mathit{Fgf4}$_GATA6', r'$\mathit{Nanog}$_A-ERK',
    'mRNA', 'PROTEIN', 'FGF4-LUMEN',
    r'$\tau_{\mathrm{d},\mathrm{FGF4}}$', r'$\tau_{\mathrm{d},\mathrm{M-FGFR-FGF4}}$', r'$\tau_{\mathrm{d},\mathrm{FGF4-LUMEN}}$', r'$\tau_{\mathrm{d},\mathrm{FGF4-SINK}}$',
    r'$\tau_{\mathrm{escape}}$', r'$\tau_{\mathrm{exchange}}$', r'$\tau_{\mathrm{LUMEN}}$', r'$\tau_{\mathrm{SINK}}$',
    r'$\tau_{\mathrm{dime},\mathrm{M-FGFR-FGF4}}$', r'$\tau_{\mathrm{mono},\mathrm{D-FGFR-FGF4}}$',
    r'$\tau_{\mathrm{pho},\mathrm{ERK}}$', r'$\tau_{\mathrm{doh},\mathrm{ERK}}$',
    r'$\tau_{\mathrm{pho},\mathrm{NANOG}}$', r'$\tau_{\mathrm{doh},\mathrm{NANOG}}$',
    r'$\chi_{\mathrm{auto}}$'
]
assert len(para_key_label_set) == len(paras), 'Oops! Something went wrong!'

para_key_elect_set = [
    ['N_N', 'G_G', 'G_N', 'N_G', 'FC_N', 'G_EA', 'FC_G', 'N_EA',
    'tau_df_FD', 'tau_db_FD', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP',
    'td_FC', 'td_FM', 'td_FL', 'td_FS', 'tau_C', 'tau_M', 'tau_L', 'tau_S',
    'MRNA', 'PRO', 'FL', 'chi_auto']
]
assert all([para in sum(para_key_elect_set, []) for para in paras]), 'Oops! Something went wrong!'

para_key_caste_set = list()
for para_key_elect in para_key_elect_set:
    para_key_caste = [paras.index(para_key) for para_key in para_key_elect]
    para_key_caste_set.append(para_key_caste)

posterior = pick_posterior
fig_size_panel = (fig_size_base, fig_size_base)
verbose = True
theta_set_opals = torch.tensor(list(_pick_mapes.values()))
theta_set_cocos = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple'] + ['tab:cyan', 'tab:olive', 'tab:pink'] + ['magenta']
para_value_set = [_[1] for _ in para_set_true.values()]
para_key_label = para_key_label_set
mark_size_alp = 3 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}

fig_rows = 1
fig_cols = 1
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = 'constrained')
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)

codas = [cellulates.index(pick[0])] # codas = [0, 1, 2, 3, 4, 5, 6, 7]
heat_cores = list()
heat_core_codas = list()
for coda in codas:
    theta_set_picks = {'post': cellulates.index(pick[0]), 'coda': coda}
    for para_key_elect_index in range(len(para_key_elect_set)):
        para_key_caste = para_key_caste_set[para_key_elect_index]
        chart, chart_coda, posterior_samples, _core_coda = synthesizer_post_coda(
            posterior = posterior, observation = None, posterior_sample_shape = tuple([250000]), parameter_set_true = para_set_true, mape_calculate = False, fig_size = fig_size_panel, verbose = verbose,
            theta_set_opals = theta_set_opals, theta_set_picks = theta_set_picks, theta_set_cocos = theta_set_cocos,
            para_value_set = para_value_set, para_key_caste = para_key_caste, para_key_elect = para_key_elect, para_key_label = para_key_label,
            fig = None, axes = None, fig_coda = None, axes_coda = None,
            mark_size = mark_size_alp
        )
        core = np.corrcoef(posterior_samples[:, para_key_caste].T)
        heat_cores.append(core)
        c_core_coda = _core_coda[:, para_key_caste] # Col Permutation
        r_c_core_coda = c_core_coda[para_key_caste, :] # Row Permutation
        core_coda = r_c_core_coda
        heat_core_codas.append(core_coda)

#%%# Color Map Test

color_map_test = plt.get_cmap(name = 'PiYG')
_color_map_zero_RGBA = color_map_test(0)
color_map_zero_RGBA = [np.round(color_map_test.N*color, 0).astype(int) for color in _color_map_zero_RGBA]
_color_map_last_RGBA = color_map_test(color_map_test.N-1)
color_map_last_RGBA = [np.round(color_map_test.N*color, 0).astype(int) for color in _color_map_last_RGBA]
color_map_zero_HSL = {'h': 325, 's': 95, 'l': 25}
color_map_last_HSL = {'h': 110, 's': 60, 'l': 25}

def make_heat_mask(heat_mate):
    heat_mask = np.invert(heat_mate.astype(bool))
    rows, cols = heat_mate.shape
    for row in range(rows):
        for col in range(cols):
            if col < row:
                heat_mask[row, col] = True
            else:
                continue
    return heat_mask

def cure_heat_mask(heat_mask):
    rows, cols = heat_mask.shape
    cure = [(index+1, index) for index in range(0, cols, 2)] # [(index+1, index) for index in range(rows)]
    for row in range(rows):
        for col in range(cols):
            if (row, col) in cure:
                heat_mask[row, col] = False
            else:
                continue
    return heat_mask

#%%# Make Heat Core Codas Plot!

from seaborn import heatmap, diverging_palette

font_size_alp = 13 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 7 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 11 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!

fig_rows = 1
fig_cols = 1
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
cmap = diverging_palette(h_neg = 330, h_pos = 110, s = 100, l = 40, sep = int(0.125*pow(2, 8))-2, n = pow(2, 8), center = 'light', as_cmap = False) # {'PiYG', 'Spectral', 'RdBu', 'coolwarm', 'bwr'}

paras = ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'MRNA', 'PRO', 'FL', 'td_FC', 'td_FM', 'td_FL', 'td_FS', 'tau_C', 'tau_M', 'tau_L', 'tau_S', 'tau_df_FD', 'tau_db_FD', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto']
cases = list(range(0, len(cellulates))) # Careful!
dire = int(len(cases)/2)
rows, cols = len(paras), len(paras)
heat_dime = (len(paras)*dire, len(paras)*dire)
heat_delts = [(0, 0), (1, 1), (0, 1), (1, 0)]
heat_mates = list() # 10/12.5
heat_masks = list() # Small Model · Large Model

tick_step = (1-0)/pow(2, 3)
ticks = np.arange(start = -1, stop = 1+tick_step, step = tick_step).tolist()
_tick_labels = [tick if ticks.index(tick) % 2 == 0 else None for tick in ticks]
tick_labels = [int(tick_label) if tick_label is not None and np.round(tick_label) == tick_label else tick_label for tick_label in _tick_labels]

para_selection = para_key_elect_set[0]
para_selection_index = [paras.index(para) for para in para_selection]
para_selection_label = [para_key_label_set[index] for index in para_selection_index]
flag = True

case = cases[0] # Careful!

fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
axe = axe_mat

heat_core = np.round(heat_cores[case], 5)
index = heat_core.shape[0] # index = heat_core.shape[0]*dire
# Special Annotation [START]
heat_pate = np.full(shape = heat_core.shape, fill_value = len(paras), dtype = np.float32)
heat_pate_mask = np.full(shape = heat_core.shape, fill_value = True, dtype = bool)
heat_pate_anna = np.full(shape = heat_core.shape, fill_value = '', dtype = 'object')
heat_pate_here = [(index-1, 1), (index-2, 0)]
annotate_heat_pate = [pick[0], sum(pick[0])]
heat_pate[heat_pate_here[0][0], heat_pate_here[0][1]] = cellulates.index(pick[0])
heat_pate[heat_pate_here[1][0], heat_pate_here[1][1]] = cellulates.index(pick[0])
heat_pate_mask[heat_pate_here[0][0], heat_pate_here[0][1]] = False
heat_pate_mask[heat_pate_here[1][0], heat_pate_here[1][1]] = False
heat_pate_anna[heat_pate_here[0][0], heat_pate_here[0][1]] = annotate_heat_pate[0]
heat_pate_anna[heat_pate_here[1][0], heat_pate_here[1][1]] = annotate_heat_pate[1]
heatmap(data = heat_pate, cmap = theta_set_cocos[:-1]+['white'], vmin = 0, vmax = len(cellulates), annot = heat_pate_anna, fmt = '', annot_kws = {'color': theta_set_cocos[cellulates.index(pick[0])], 'fontsize': font_size_alp, 'fontweight': 'bold', 'rotation': 45}, cbar = False, square = True, xticklabels = [], yticklabels = [], ax = axe[0, 0], alpha = 0.25)
# Special Annotation [FINAL]
heat_mask = make_heat_mask(heat_core)
heatmap(data = heat_core, mask = heat_mask, cmap = cmap, vmin = -1, vmax = 1, linewidths = 0, linecolor = 'white', annot = True, fmt = '.2f', annot_kws = {'fontsize': font_size_chi-3}, cbar = flag, square = True, xticklabels = [], yticklabels = [], ax = axe[0, 0], cbar_kws = {'ticks': None, 'location': 'bottom', 'fraction': 0.05, 'pad': 0.005})
heat_tick_step = 1
heat_ticks = np.arange(start = 0.5, stop = len(paras), step = heat_tick_step).tolist()
axe[0, 0].tick_params(bottom = False, top = True, left = False, right = True, labelbottom = False, labeltop = True, labelleft = False, labelright = True)
axe[0, 0].set_xticks(ticks = heat_ticks, labels = para_selection_label, fontsize = font_size_bet, rotation = 20, rotation_mode = 'anchor', horizontalalignment = 'left', verticalalignment = 'center')
axe[0, 0].set_yticks(ticks = heat_ticks, labels = para_selection_label, fontsize = font_size_bet, rotation = 270+20, rotation_mode = 'anchor', horizontalalignment = 'left', verticalalignment = 'center')
axe[0, 0].vlines(x = np.arange(0, len(paras)+1, 1), ymin = 0, ymax = len(paras)+1, colors = 'w', linewidths = 2, alpha = 1)
axe[0, 0].hlines(y = np.arange(0, len(paras)+1, 1), xmin = 0, xmax = len(paras)+1, colors = 'w', linewidths = 2, alpha = 1)
axe[0, 0].figure.axes[-1].set_xticks(ticks = ticks, labels = tick_labels, fontsize = font_size_chi)
axe[0, 0].set_title(label = 'Unconditional Correlation Coefficient', y = -0.025, pad = 0, fontsize = font_size_alp) # "Unconditional Pearson's Product-Moment Correlation Coefficient"
plt.show()

fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)
axe = axe_mat

heat_core_coda = np.round(heat_core_codas[case].numpy(), 5)
index = heat_core_coda.shape[0] # index = heat_core.shape[0]*dire
# Special Annotation [START]
heat_pate = np.full(shape = heat_core_coda.shape, fill_value = len(paras), dtype = np.float32)
heat_pate_mask = np.full(shape = heat_core_coda.shape, fill_value = True, dtype = bool)
heat_pate_anna = np.full(shape = heat_core_coda.shape, fill_value = '', dtype = 'object')
heat_pate_here = [(index-1, 1), (index-2, 0)]
annotate_heat_pate = [pick[0], sum(pick[0])]
heat_pate[heat_pate_here[0][0], heat_pate_here[0][1]] = cellulates.index(pick[0])
heat_pate[heat_pate_here[1][0], heat_pate_here[1][1]] = cellulates.index(pick[0])
heat_pate_mask[heat_pate_here[0][0], heat_pate_here[0][1]] = False
heat_pate_mask[heat_pate_here[1][0], heat_pate_here[1][1]] = False
heat_pate_anna[heat_pate_here[0][0], heat_pate_here[0][1]] = annotate_heat_pate[0]
heat_pate_anna[heat_pate_here[1][0], heat_pate_here[1][1]] = annotate_heat_pate[1]
heatmap(data = heat_pate, cmap = theta_set_cocos[:-1]+['white'], vmin = 0, vmax = len(cellulates), annot = heat_pate_anna, fmt = '', annot_kws = {'color': theta_set_cocos[cellulates.index(pick[0])], 'fontsize': font_size_alp, 'fontweight': 'bold', 'rotation': 45}, cbar = False, square = True, xticklabels = [], yticklabels = [], ax = axe[0, 0], alpha = 0.25)
# Special Annotation [FINAL]
heat_mask = make_heat_mask(heat_core_coda)
heatmap(data = heat_core_coda, mask = heat_mask, cmap = cmap, vmin = -1, vmax = 1, linewidths = 0, linecolor = 'white', annot = True, fmt = '.2f', annot_kws = {'fontsize': font_size_chi-3}, cbar = flag, square = True, xticklabels = [], yticklabels = [], ax = axe[0, 0], cbar_kws = {'ticks': None, 'location': 'bottom', 'fraction': 0.05, 'pad': 0.005})
heat_tick_step = 1
heat_ticks = np.arange(start = 0.5, stop = len(paras), step = heat_tick_step).tolist()
axe[0, 0].tick_params(bottom = False, top = True, left = False, right = True, labelbottom = False, labeltop = True, labelleft = False, labelright = True)
axe[0, 0].set_xticks(ticks = heat_ticks, labels = para_selection_label, fontsize = font_size_bet, rotation = 20, rotation_mode = 'anchor', horizontalalignment = 'left', verticalalignment = 'center')
axe[0, 0].set_yticks(ticks = heat_ticks, labels = para_selection_label, fontsize = font_size_bet, rotation = 270+20, rotation_mode = 'anchor', horizontalalignment = 'left', verticalalignment = 'center')
axe[0, 0].vlines(x = np.arange(0, len(paras)+1, 1), ymin = 0, ymax = len(paras)+1, colors = 'w', linewidths = 2, alpha = 1)
axe[0, 0].hlines(y = np.arange(0, len(paras)+1, 1), xmin = 0, xmax = len(paras)+1, colors = 'w', linewidths = 2, alpha = 1)
axe[0, 0].figure.axes[-1].set_xticks(ticks = ticks, labels = tick_labels, fontsize = font_size_chi)
axe[0, 0].set_title(label = 'Conditional Correlation Coefficient', y = -0.025, pad = 0, fontsize = font_size_alp) # "Conditional Pearson's Product-Moment Correlation Coefficient"
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig6'
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
