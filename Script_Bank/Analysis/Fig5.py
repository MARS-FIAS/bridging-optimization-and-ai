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

#%%# Retrieve Posterior Data (AI_L1)

from Fig0_Services import retrieve_AI_L1_posterior_mape, make_paras

AI_L1_posterior_mape = retrieve_AI_L1_posterior_mape()
AI_L1_posterior, _AI_L1_mape = AI_L1_posterior_mape[0][0], AI_L1_posterior_mape[1][0]

AI_L1_para_set_raw = {
    'N_N': (100, (0, 1000)), 'G_G': (200, (0, 1000)), 'FC_N': (500, (0, 1000)), 'G_EA': (750, (0, 1000)),
    'G_N': (50, (0, 1000)), 'N_G': (400, (0, 1000)), 'FC_G': (500, (0, 1000)), 'N_EA': (750, (0, 1000)),
    'MRNA': (50, (0, 250)), 'PRO': (200, (0, 1000)),
    'td_FC': (7200, (300, 28800)), 'td_FM': (7200, (300, 28800)),
    'tau_C': (450, (30, 5*900)), 'tau_M': (450, (30, 2*2100)),
    'tau_ef_EA': (17100, (10*30, 43200)), 'tau_eb_EA': (300, (30, 43200)), 'tau_pf_NP': (1710, (10*30, 43200)), 'tau_pb_NP': (171, (30, 43200)),
    'chi_auto': (0.5, (0, 1))
}
AI_L1_para_set_mode = 0 # {0, 1}
_AI_L1_para_set, _AI_L1_para_set_true = make_paras(para_set_raw = AI_L1_para_set_raw, para_set_mode = AI_L1_para_set_mode, verbose = True)
_AI_L1_para_keys = list(_AI_L1_para_set_true.keys())
_AI_L1_para_values = list(_AI_L1_para_set_true.values())
AI_L1_para_set_true = {key: None for key in para_set_true.keys()}
lulu_paras = ['tau_df_FD', 'tau_db_FD']
for key in para_set_true:
    if key in _AI_L1_para_keys:
        AI_L1_para_index = _AI_L1_para_keys.index(key)
        AI_L1_para_value = _AI_L1_mape[AI_L1_para_index]
        AI_L1_para_span = _AI_L1_para_values[AI_L1_para_index][1]
        AI_L1_para_key = key
        AI_L1_para_set_true[AI_L1_para_key] = (AI_L1_para_value, AI_L1_para_span)
    elif key in lulu_paras:
        AI_L1_para_set_true[key] = para_set_true[key]
    else:
        AI_L1_para_set_true[key] = (np.nan, (np.nan, np.nan))

#%%# Extract Posterior Sample Mat AND MAPE Vet (AI_L1)

AI_L1_posterior_samples_temp, _AI_L1_theta = synthesizer_post(posterior = AI_L1_posterior, observation = None, posterior_sample_shape = tuple([250000]))
AI_L1_posterior_samples = np.full(shape = pick_posterior_samples.shape, fill_value = np.nan, dtype = np.float32)
AI_L1_mape_temp = np.array([_[0] for _ in AI_L1_para_set_true.values()])
AI_L1_mape = np.full(shape = pick_theta.shape, fill_value = np.nan, dtype = np.float32)
AI_L1_nork_paras = [AI_L1_para_set_true[paras[_]][1][1]-AI_L1_para_set_true[paras[_]][1][0] for _ in range(0, len(paras))]
tally = 0
for key in para_set_true:
    if key in _AI_L1_para_keys:
        AI_L1_para_index = _AI_L1_para_keys.index(key)
        AI_L1_posterior_samples[:, tally] = AI_L1_posterior_samples_temp[:, AI_L1_para_index]/AI_L1_nork_paras[tally]
        AI_L1_mape[tally] = AI_L1_mape_temp[tally]/AI_L1_nork_paras[tally]
    elif key in lulu_paras:
        AI_L1_mape[tally] = AI_L1_mape_temp[tally]/AI_L1_nork_paras[tally]
    else:
        pass
    tally += 1

#%%# Compare Parameter Value (MAPE!) Sets [Separate] Plot!

from seaborn import boxplot

para_labels = [
    r'$\mathit{Nanog}$_NANOG', r'$\mathit{Gata6}$_GATA6', r'$\mathit{Fgf4}$_NANOG', r'$\mathit{Gata6}$_A-ERK', r'$\mathit{Gata6}$_NANOG', r'$\mathit{Nanog}$_GATA6', r'$\mathit{Fgf4}$_GATA6', r'$\mathit{Nanog}$_A-ERK',
    'mRNA', 'PROTEIN', 'FGF4-LUMEN',
    r'$\tau_{\mathrm{d},\mathrm{FGF4}}$', r'$\tau_{\mathrm{d},\mathrm{M-FGFR-FGF4}}$', r'$\tau_{\mathrm{d},\mathrm{FGF4-LUMEN}}$', r'$\tau_{\mathrm{d},\mathrm{FGF4-SINK}}$',
    r'$\tau_{\mathrm{escape}}$', r'$\tau_{\mathrm{exchange}}$', r'$\tau_{\mathrm{LUMEN}}$', r'$\tau_{\mathrm{SINK}}$',
    r'$\tau_{\mathrm{dime},\mathrm{M-FGFR-FGF4}}$', r'$\tau_{\mathrm{mono},\mathrm{D-FGFR-FGF4}}$',
    r'$\tau_{\mathrm{pho},\mathrm{ERK}}$', r'$\tau_{\mathrm{doh},\mathrm{ERK}}$',
    r'$\tau_{\mathrm{pho},\mathrm{NANOG}}$', r'$\tau_{\mathrm{doh},\mathrm{NANOG}}$',
    r'$\chi_{\mathrm{auto}}$'
]
assert len(para_labels) == len(paras), 'Oops! Something went wrong!'

para_selections = [
    ['N_N', 'G_G', 'G_N', 'N_G', 'FC_N', 'G_EA', 'FC_G', 'N_EA'],
    ['tau_df_FD', 'tau_db_FD', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP'],
    ['td_FC', 'td_FM', 'td_FL', 'td_FS', 'tau_C', 'tau_M', 'tau_L', 'tau_S'],
    ['MRNA', 'PRO', 'FL', 'chi_auto']
]
assert all([para in sum(para_selections, []) for para in paras]), 'Oops! Something went wrong!'

font_size_alp = 29 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!
almas = [0, 0.25, 0.5, 0.75, 1]
lisps = ['', '-', '--', ':', '-.']
lites = [1, 2, 3, 5, 7]
marks = ['', '.', 'o', 'D', 'X']
cocos = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple'] + ['tab:cyan', 'tab:olive', 'tab:pink']

boxprops = {'edgecolor': 'tab:gray', 'linewidth': lites[1], 'alpha': almas[1]}
capprops = {'color': 'tab:gray', 'linewidth': lites[1], 'alpha': almas[3]}
medianprops = {'color': 'tab:gray', 'linewidth': lites[3], 'alpha': almas[3]}
whiskerprops = {'color': 'tab:gray', 'linewidth': lites[1], 'alpha': almas[3]}

capwidths = 0.125
whisk = (2.5, 97.5) # [0, 100]

y_limes = (0, 1, 0.05)
y_ticks = np.round(np.arange(y_limes[0], y_limes[1]+y_limes[2], y_limes[2]), 2)
y_tick_labels = [f'{int(100*y_tick)}%' if (10*y_tick) % (10*2*y_limes[2]) == 0 else None for y_tick in y_ticks]

fig_rows = 2
fig_cols = 2
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = 'constrained')
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)

AI_L1_coco = 'tab:red'
AI_L1_label = 'AI_L1'
widen = 0.25
delta = widen/2

for para_selection in para_selections:
    
    fig_lex = para_selections.index(para_selection)
    fig_row = fig_lex // fig_rows
    fig_col = fig_lex % fig_cols
    pick_coco = cocos[cellulates.index(pick[0])]
    
    para_selection_index = [paras.index(para) for para in para_selection]
    para_selection_label = [para_labels[_] for _ in para_selection_index]
    lulu_selection_index = [paras.index(para) for para in lulu_paras]
    
    x_limes = (-0.5, len(para_selection)-0.5)
    x_ticks = list(range(0, len(para_selection)))
    x_tick_labels = para_selection_label
    
    alp = pick_posterior_samples[:, para_selection_index]
    boxprops['facecolor'] = pick_coco
    positions = [x_tick-delta for x_tick in x_ticks]
    axe_mat[fig_row, fig_col].boxplot(x = alp, positions = positions, widths = widen, patch_artist = True, whis = whisk, showfliers = False, boxprops = boxprops, capprops = capprops, medianprops = medianprops, whiskerprops = whiskerprops, zorder = 1, capwidths = capwidths)
    bet = AI_L1_posterior_samples[:, para_selection_index]
    boxprops['facecolor'] = AI_L1_coco
    positions = [x_tick+delta for x_tick in x_ticks]
    axe_mat[fig_row, fig_col].boxplot(x = bet, positions = positions, widths = widen, patch_artist = True, whis = whisk, showfliers = False, boxprops = boxprops, capprops = capprops, medianprops = medianprops, whiskerprops = whiskerprops, zorder = 1, capwidths = capwidths)
    
    for cellulate in [pick[0]]:
        print(f"Cellulate{' '*4}{cellulate}")
        cellulate_index = cellulates.index(cellulate)
        size = sum(cellulate)
        x = np.arange(0, len(para_selection))
        y = pick_mapes[cellulate][para_selection_index]
        coco = cocos[cellulates.index(cellulate)]
        axe_mat[fig_row, fig_col].plot(x, y, marker = marks[1], markerfacecolor = coco, markersize = font_size_bet, linestyle = lisps[1], linewidth = lites[2], color = coco, label = f'{cellulate} ~ {size}' if fig_lex == 0 else None, alpha = almas[4])
        axe_mat[fig_row, fig_col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet if fig_lex not in [0, 3] else font_size_ate, rotation = 30)
        axe_mat[fig_row, fig_col].set_yticks(ticks = y_ticks, labels = y_tick_labels if fig_col == 0 else [], fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].set_xlim(x_limes[0], x_limes[1])
        axe_mat[fig_row, fig_col].set_ylim(y_limes[0], y_limes[1])
        axe_mat[fig_row, fig_col].grid(axis = 'y', linestyle = lisps[3], color = 'tab:gray', alpha = almas[2])
    exam = any([lulu in para_selection_index for lulu in lulu_selection_index])
    if exam:
        _x = [para_selection_index.index(lulu) for lulu in lulu_selection_index]
        _z = AI_L1_mape[lulu_selection_index]
        axe_mat[fig_row, fig_col].plot(_x, _z, marker = marks[3], markerfacecolor = 'lime', markersize = font_size_alp, linestyle = lisps[0], linewidth = lites[3], color = AI_L1_coco, label = AI_L1_label if fig_lex == 0 else None, alpha = almas[4])
        axe_mat[fig_row, fig_col].plot(_x, _z, marker = marks[1], markerfacecolor = AI_L1_coco, markersize = font_size_bet, linestyle = lisps[1], linewidth = lites[2], color = AI_L1_coco, label = AI_L1_label if fig_lex == 0 else None, alpha = almas[4])
        z = AI_L1_mape[para_selection_index]
        z[_x] = np.nan
    else:
        z = AI_L1_mape[para_selection_index]
    axe_mat[fig_row, fig_col].plot(x, z, marker = marks[1], markerfacecolor = AI_L1_coco, markersize = font_size_bet, linestyle = lisps[1], linewidth = lites[2], color = AI_L1_coco, label = AI_L1_label if fig_lex == 0 else None, alpha = almas[4])
    
    if fig_col == 0: axe_mat[fig_row, fig_col].set_ylabel(ylabel = 'Parameter Value [Prior Range]', fontsize = font_size_bet)

fig.legend(loc = 'center', bbox_to_anchor = (0.2675, 0.4875), ncols = 4, fontsize = font_size_alp, edgecolor = 'magenta', borderpad = 0.125, labelspacing = 0.25, columnspacing = 1)
fig.suptitle(t = 'Parameter Value Set (Comparison)\n[Posterior Range]', fontsize = font_size_alp)
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig5'
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
