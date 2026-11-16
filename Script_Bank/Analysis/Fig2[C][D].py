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

#%%# Meta-Score Round Progession

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 19 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_ate = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} # Annotate!

almas = [0.5, 0.75]
lisps = ['', '-']
lites = [2, 3]
marks = ['', '.']

cocos = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple'] + ['tab:cyan', 'tab:olive', 'tab:pink']
annotation = '\n\n'.join(['Meta Score', r'$\overline{S} = \overline{\alpha}_{75} + \overline{\alpha}_{25} - \overline{\alpha}_{50}$'])

#%%# Meta-Score Round Progession Alp

fig_rows = 1
fig_cols = 1

fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = 'constrained')
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)

fig_row = 0
fig_col = 0
x = np.array(object = acts, dtype = int)+1
x_ticks = x
y_ticks = np.round(np.linspace(0, 1, 11), 1)

x_label = 'Round'
y_label = 'Meta Score'
tit = 'Meta-Score Round Progression'
panel = '[C]'

for cellulate in cellulates:
    if len(cellulate) != 2:
        continue
    else:
        print(f"Cellulate{' '*4}{cellulate}")
        cellulate_index = cellulates.index(cellulate)
        size = sum(cellulate)
        posteriors = posterior_data[cellulate]['posteriors']
        appraisals = [posterior.appraisal for posterior in posteriors]
        y = np.array(appraisals)
        coco = cocos[cellulate_index]
        mark = marks[1]
        lite = lites[1]
        lisp = lisps[1]
        alma = almas[1]
        axe_mat[fig_row, fig_col].plot(x, y, color = coco, marker = mark, linewidth = lite, linestyle = lisp, alpha = alma, markersize = font_size_bet, label = f'{cellulate} ~ {size}')
        axe_mat[fig_row, fig_col].set_title(label = tit, loc = 'center', pad = 11, fontsize = font_size_alp, fontweight = 'normal')
        axe_mat[fig_row, fig_col].set_title(label = panel, loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold')
        axe_mat[fig_row, fig_col].set_xlim(np.min(x)-0.25, np.max(x)+0.25)
        axe_mat[fig_row, fig_col].set_ylim(0-0.025, 1+0.025)
        axe_mat[fig_row, fig_col].set_xticks(ticks = x_ticks, labels = x_ticks, fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].set_yticks(ticks = y_ticks, labels = y_ticks, fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].set_xlabel(xlabel = x_label, fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].set_ylabel(ylabel = y_label, fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].grid(axis = 'both', linestyle = '--', color = 'tab:gray', alpha = 0.25)
        axe_mat[fig_row, fig_col].legend(loc = 'center right', ncols = 1, fontsize = font_size_chi, title = 'Shape ~ Size', title_fontsize = font_size_chi)
        axe_mat[fig_row, fig_col].annotate(text = annotation, xy = (4.5, 0.075), color = 'black', fontsize = font_size_ate, fontweight = 'bold', ha = 'center', va = 'center')
plt.show()

#%%# Meta-Score Round Progession Bet

fig_rows = 1
fig_cols = 1
fig_size = (fig_cols*fig_size_base, fig_rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = 'constrained')
axe_mat = fig.subplots(nrows = fig_rows, ncols = fig_cols, squeeze = False)

fig_row = 0
fig_col = 0
x = np.array(object = acts, dtype = int)+1
x_ticks = x
y_ticks = np.round(np.linspace(0, 1, 11), 1)

x_label = 'Round'
y_label = 'Meta Score'
tit = 'Meta-Score Round Progression'
panel = '[D]'

for cellulate in cellulates:
    if len(cellulate) == 2:
        continue
    else:
        print(f"Cellulate{' '*4}{cellulate}")
        cellulate_index = cellulates.index(cellulate)
        size = sum(cellulate)
        posteriors = posterior_data[cellulate]['posteriors']
        appraisals = [posterior.appraisal for posterior in posteriors]
        y = np.array(appraisals)
        coco = cocos[cellulate_index]
        mark = marks[1]
        lite = lites[1]
        lisp = lisps[1]
        alma = almas[1]
        axe_mat[fig_row, fig_col].plot(x, y, color = coco, marker = mark, linewidth = lite, linestyle = lisp, alpha = alma, markersize = font_size_bet, label = f'{cellulate} ~ {size}')
        axe_mat[fig_row, fig_col].set_title(label = tit, loc = 'center', pad = 11, fontsize = font_size_alp, fontweight = 'normal')
        axe_mat[fig_row, fig_col].set_title(label = panel, loc = 'left', pad = 11, fontsize = font_size_alp, fontweight = 'bold')
        axe_mat[fig_row, fig_col].set_xlim(np.min(x)-0.25, np.max(x)+0.25)
        axe_mat[fig_row, fig_col].set_ylim(0-0.025, 1+0.025)
        axe_mat[fig_row, fig_col].set_xticks(ticks = x_ticks, labels = x_ticks, fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].set_yticks(ticks = y_ticks, labels = y_ticks, fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].set_xlabel(xlabel = x_label, fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].set_ylabel(ylabel = y_label, fontsize = font_size_bet)
        axe_mat[fig_row, fig_col].grid(axis = 'both', linestyle = '--', color = 'tab:gray', alpha = 0.25)
        axe_mat[fig_row, fig_col].legend(loc = 'center right', ncols = 1, fontsize = font_size_chi, title = 'Shape ~ Size', title_fontsize = font_size_chi)
        axe_mat[fig_row, fig_col].annotate(text = annotation, xy = (4.5, 0.075), color = 'black', fontsize = font_size_ate, fontweight = 'bold', ha = 'center', va = 'center')
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig2[C][D]'
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
