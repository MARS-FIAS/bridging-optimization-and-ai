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
import numpy as np
import torch
import pickle
import time
import re

#%%# Retrieve Data [Cellulate]

def posterior_appraisal_selection(path, _tag, postage = None, verbose = False):
    if postage is None:
        _arcs = os.listdir(path)
        arcs = [arc for arc in _arcs if re.findall(_tag+'_'+'Posterior', arc)]
        tasks = [int(re.findall('(\d+)\.pkl', arc)[0]) for arc in arcs if re.findall('(\d+)\.pkl', arc)]
        tasks.sort()
        if len(tasks) == 0 and len(arcs) == 0:
            mess = 'Oops! Where are the posteriors?'
            raise RuntimeError(mess)
        elif len(tasks) == 0 and len(arcs) != 0:
            _postage = '_Posterior.pkl'
            with open(path + _tag + _postage, 'rb') as portfolio:
                posterior = pickle.load(portfolio)
            mess = f"{'~'*4*8}\nPosterior!\n{' '*8}{path + _tag + _postage}\n{'~'*4*8}"
        else:
            posts = list()
            appraisals = list()
            for task in tasks:
                _postage = f'_Posterior_{task}.pkl'
                with open(path + _tag + _postage, 'rb') as portfolio:
                    post = pickle.load(portfolio)
                appraisal = post.appraisal if hasattr(post, 'appraisal') else 0
                posts.append(post)
                appraisals.append(appraisal)
            where = np.argmax(appraisals)
            posterior = posts[where]
            _mess = f'{path + _tag}_Posterior_{where}.pkl'
            mess = f"{'~'*4*8}\nPosterior Tasks!\n{' '*8}{tasks}\nPosterior Appraisals!\n{' '*8}{appraisals}\nPosterior Where?\n{' '*8}{where}\nPosterior!\n{' '*8}{_mess}\n{'~'*4*8}"
    else:
        with open(path + _tag + postage, 'rb') as portfolio:
            posterior = pickle.load(portfolio)
        mess = f"{'~'*4*8}\nPosterior!\n{' '*8}{path + _tag + postage}\n{'~'*4*8}"
    if verbose: print(mess)
    return posterior

def retrieve_posteriors_mapes(data_paths, acts, observers, curbs, verbose = False):
    if verbose:
        tip = 's' if len(data_paths) > 1 else ''
        print(f"Info!\n\tWe will retrieve {len(data_paths)} {'posterior' + tip} and {len(data_paths)} {'mape' + tip}!")
    posteriors = list()
    mapes = list()
    for index in range(len(data_paths)):
        data_path = data_paths[index]
        act = acts[index]
        observe = observers[index]
        curb = curbs[index]
        path = f'REFERENCE_PANEL_DATA/{data_path}/Observe_{observe}/'
        _tag = f'Act_{act}_Observe_{observe}_Cellulate_{curb}'
        postage = None # postage = '_Posterior.pkl'
        posterior = posterior_appraisal_selection(path, _tag, postage, verbose = True)
        _mape = posterior.map().numpy()
        mape = [int(round(_mape[index], 0)) if _mape[index] > 1 else round(_mape[index], 2) for index in range(_mape.size)]
        if verbose: print(f'Posterior MAPE!\n{data_path}\n{_tag}\n\t{mape}')
        posteriors.append(posterior)
        mapes.append(mape)
        ret = (posteriors, mapes)
    return ret

def retrieve_AI_L1_posterior_mape(data_paths = ['Shallow_Grid_1_N_Link'], acts = [7], observers = [1], curbs = ['Mid'], verbose = False):
    if verbose:
        tip = 's' if len(data_paths) > 1 else ''
        print(f"Info!\n\tWe will retrieve {len(data_paths)} {'posterior' + tip} and {len(data_paths)} {'mape' + tip}!")
    posteriors = list()
    mapes = list()
    for index in range(len(data_paths)):
        data_path = data_paths[index]
        act = acts[index]
        observe = observers[index]
        curb = curbs[index]
        path = f'REFERENCE_EARLIER_MODEL_DATA/{data_path}/Observe_{observe}/'
        _tag = f'Act_{act}_Observe_{observe}_{curb}'
        postage = None # postage = '_Posterior.pkl'
        posterior = posterior_appraisal_selection(path, _tag, postage, verbose = True)
        _mape = posterior.map().numpy()
        mape = [int(round(_mape[index], 0)) if _mape[index] > 1 else round(_mape[index], 2) for index in range(_mape.size)]
        if verbose: print(f'Posterior MAPE!\n{data_path}\n{_tag}\n\t{mape}')
        posteriors.append(posterior)
        mapes.append(mape)
        ret = (posteriors, mapes)
    return ret

def make_paras(para_set_raw = None, para_set_mode = None, verbose = False):
    descript = "{'Parameter': ('Act_Value', ('Mini_Value', 'Maxi_Value')), 'Mode': {0, 1}}"
    if verbose: print(descript)
    if para_set_raw is not None:
        mess = "The variable 'para_set_raw' must be a valid dictionary!"
        check = type(para_set_raw) is dict
        assert check, mess
        mess = 'Every dictionary key must be a string!'
        _check = [type(key) is str for key in para_set_raw.keys()]
        check = all(_check)
        assert check, mess
        mess = 'Every dictionary value must be a duple!'
        _check = [type(value) is tuple and len(value) == 2 for value in para_set_raw.values()]
        check = all(_check)
        assert check, mess
        mess = "Every duple must specify a valid 'true' parameter value and a valid 'duple' parameter range!"
        _check = [value[1][0] <= value[0] <= value[1][1] for value in para_set_raw.values()]
        check = all(_check)
        assert check, mess
        para_set_true = para_set_raw
        if para_set_mode == 0:
            para_set = np.array([value[0] for value in para_set_true.values()])
        elif para_set_mode == 1:
            para_set = np.array([(value[0]-value[1][0])/(value[1][1]-value[1][0]) for value in para_set_true.values()])
        else:
            mess = f"Invalid mode!\n\t'Mode = {para_set_mode}'"
            raise RuntimeError(mess)
    else:
        para_set_true = None
        para_set = None
    paras = (para_set, para_set_true)
    return paras

def geometric_median(mate_zero, paras, para_set_true, capacity = 50e3):
    from scipy.spatial.distance import pdist, cdist, squareform
    mate = np.full(shape = mate_zero.shape, fill_value = np.nan)
    for i in range(0, len(paras)):
        mate[:, i] = mate_zero[:, i]/(para_set_true[paras[i]][1][1]-para_set_true[paras[i]][1][0])
    if mate.shape[0] < capacity:
        _mate_dist = pdist(mate)
        mate_dist = squareform(_mate_dist)
        mate_suma = np.sum(mate_dist, 0)
        print('Matrix!')
    else:
        mate_suma = np.zeros(shape = tuple([mate.shape[0]]))
        for j in range(mate.shape[0]):
            mate_cure = mate[j, :].reshape((1, -1))
            mate_dist = cdist(mate_cure, mate)
            mate_suma[j] = np.sum(mate_dist)
        print('Loop!')
    _mate_para = np.isclose(a = mate_suma, b = np.min(mate_suma), atol = 1e-13)
    mate_para = mate_zero[_mate_para]
    if len(mate_para) == 0:
        mess = 'Oops! Something went wrong!'
        raise RuntimeError(mess)
    elif len(mate_para) > 1:
        print('Oops!\n\tWe must choose one parameter set!\n\tWe take the parameter set at position ZERO!')
    ret = mate_para[0, :]
    return ret

#%%# Posterior Synthesizer

from sbi import analysis

def synthesizer_post(posterior = None, observation = None, posterior_sample_shape = None, parameter_set_true = None, mape_calculate = False, fig_size = (5, 5), verbose = False, **keywords):
    if posterior is None:
        mess = 'Oops! Please, if we wish to synthesize some simulator samples, then we need a posterior distribution!'
        raise RuntimeError(mess)
    check = isinstance(posterior.map(), torch.Tensor)
    mess = "Please, we must provide a valid 'posterior' object!\First, we must execute 'InferenceProd.synthesizer'!"
    assert check, mess
    if observation is None:
        observation = posterior.default_x
    posterior_samples = keywords.get('posterior_samples', None)
    if posterior_samples is None:
        posterior_samples = posterior.sample(sample_shape = posterior_sample_shape, x = observation)
    theta_median = torch.quantile(posterior_samples, 0.5, 0)
    theta_mape = posterior.map()
    theta = {'median': theta_median, 'mape': theta_mape}
    ret = (posterior_samples, theta)
    return ret

def synthesizer_post_coda(posterior = None, observation = None, posterior_sample_shape = None, parameter_set_true = None, mape_calculate = False, fig_size = (5, 5), verbose = False, **keywords):
    if posterior is None:
        mess = 'Oops! Please, if we wish to synthesize some simulator samples, then we need a posterior distribution!'
        raise RuntimeError(mess)
    check = isinstance(posterior.map(), torch.Tensor)
    mess = "Please, we must provide a valid 'posterior' object!\First, we must execute 'InferenceProd.synthesizer'!"
    assert check, mess
    paras = list(parameter_set_true.keys())
    para_span = torch.tensor([0, 1])
    if observation is None:
        observation = posterior.default_x
    posterior_samples = keywords.get('posterior_samples', None)
    if posterior_samples is None:
        posterior_samples = posterior.sample(sample_shape = posterior_sample_shape, x = observation)
    card = len(paras)
    theta_median = torch.quantile(posterior_samples, 0.5, 0)
    if mape_calculate:
        posterior.set_default_x(observation)
        posterior.map(num_init_samples = posterior_sample_shape[0])
    theta_mape = posterior.map()
    theta = {'median': theta_median, 'mape': theta_mape}
    minis = torch.floor(torch.min(posterior_samples, 0).values)
    maxis = torch.ceil(torch.max(posterior_samples, 0).values)
    check = torch.tensor([para_span[0] <= minis[index] and para_span[1] >= maxis[index] for index in range(posterior_samples.shape[1])])
    para_key_caste = keywords.get('para_key_caste', ...)
    para_key_elect = keywords.get('para_key_elect', None)
    para_key_label = keywords.get('para_key_label', paras) # para_key_label = [f"{int(np.round(theta['mape'].tolist()[index], 0)) if theta['mape'][index] >= 1 else np.round(theta['mape'].tolist()[index], 2)}\n{_para_key_label[index]}" for index in range(posterior_samples.shape[1])]
    if verbose:
        print(f'Caste\n\t{para_key_caste}\nElect\n\t{para_key_elect}\nLabel\n\t{[para_key_label[para_key_index] for para_key_index in para_key_caste]}')
    if torch.all(check):
        limes = [para_span.tolist()]*card
    else:
        para_value_set = keywords.get('para_value_set', None)
        if para_value_set is None:
            limes = torch.tensor([[minis[index], maxis[index]] for index in range(posterior_samples.shape[1])])
        else:
            limes = torch.tensor([[para_value_set[index][0], para_value_set[index][1]] for index in range(posterior_samples.shape[1])])
    spots = [[lime[0], lime[1], (lime[1]+lime[0])/2] for lime in limes]
    fig = keywords.get('fig', None)
    axes = keywords.get('axes', None)
    mark_size = keywords.get('mark_size', 7)
    theta_set_opals = keywords.get('theta_set_opals', posterior.sample())
    theta_set_picks = keywords.get('theta_set_picks', {'post': 0, 'coda': 1})
    theta_set_cocos = keywords.get('theta_set_cocos', ['magenta']*len(theta_set_opals))
    pick_post = theta_set_picks.get('post')
    pick_coda = theta_set_picks.get('coda')
    points = theta_set_opals.tolist()
    points_colors = theta_set_cocos
    chart = analysis.pairplot(
        samples = posterior_samples, points = points, limits = limes, subset = para_key_caste, offdiag = 'hist', diag = 'hist', figsize = fig_size, labels = para_key_label, ticks = spots, fig = fig, axes = axes,
        samples_colors = theta_set_cocos[slice(pick_post, pick_post+1)], points_colors = points_colors, points_diag = {'linewidth': 2}, points_offdiag = {'markersize': mark_size}, hist_diag = {'histtype': 'stepfilled', 'bins': 100, 'alpha': 0.5}, hist_offdiag = {'bins': 100}
    )
    if pick_coda != len(theta_set_opals):
        condition = theta_set_opals[[pick_coda], :] # posterior.sample() # torch.reshape(theta_mape, (1, theta_mape.size()[0]))
        points_coda = points[slice(pick_coda, pick_coda+1)]
    else:
        condition = torch.reshape(torch.mean(torch.as_tensor(limes, dtype = torch.float), dim = 1, keepdim = True), (1, len(limes)))
        points_coda = condition
    fig_coda = keywords.get('fig_coda', None)
    axes_coda = keywords.get('axes_coda', None)
    chart_coda = analysis.conditional_pairplot(
        density = posterior, condition = condition, limits = limes, points = points_coda, subset = para_key_caste, resolution = 250, figsize = fig_size, labels = para_key_label, ticks = spots, points_colors = points_colors[slice(pick_coda, pick_coda+1)], fig = fig_coda, axes = axes_coda,
        samples_colors = ['tab:gray'], points_diag = {'linewidth': 2}, points_offdiag = {'markersize': mark_size}, kde_diag = {'bins': 100}, hist_offdiag = {'bins': 100}
    )
    core_coda = analysis.conditional_corrcoeff(density = posterior, limits = limes, condition = condition, subset = np.sort(para_key_caste), resolution = 250)
    if verbose:
        print(posterior)
        print(f"MAPE\n\t{theta['mape'][para_key_caste]}")
        print(f'Condition\n\t{condition[:, para_key_caste]}')
        print(f'\n\t{para_key_elect}')
    ret = (chart, chart_coda, posterior_samples, core_coda)
    return ret

def cure_axe_mat(axe_mat = None, para_key_caste = None, para_value_set = None, mape = None, **keywords):
    cure = keywords.get('cure', set())
    verbose = keywords.get('verbose', False)
    axe_mat_rows, axe_mat_cols = axe_mat.shape
    _ave = [(para_value[1]+para_value[0])/2 for para_value in para_value_set]
    ave = [int(np.round(para_value, 0)) if para_value > 1 else np.round(para_value, 2) for para_value in _ave]
    if 'tick' in cure:
        font_size_tick = keywords.get('font_size_tick', 13)
        for axe_mat_row in range(axe_mat_rows):
            for axe_mat_col in range(axe_mat_cols):
                if axe_mat_row != axe_mat_col:
                    continue
                else:
                    axe_mat_index = axe_mat_row # axe_mat_index = axe_mat_col
                    axe_mat_caste = para_key_caste[axe_mat_index]
                    spots = [para_value_set[axe_mat_caste][0], ave[axe_mat_caste], para_value_set[axe_mat_caste][1]]
                    axe_mat[axe_mat_row, axe_mat_col].set_xticks(ticks = spots, labels = spots, fontsize = font_size_tick)
    if 'label' in cure:
        font_size_label = keywords.get('font_size_label', 17)
        for axe_mat_row in range(axe_mat_rows):
            for axe_mat_col in range(axe_mat_cols):
                if axe_mat_col > axe_mat_row:
                    axe_mat_index = axe_mat_col # axe_mat_index = axe_mat_row
                    axe_mat_caste = para_key_caste[axe_mat_index]
                    x_label = mape[axe_mat_caste]
                    y_label = mape[para_key_caste[axe_mat_row]]
                    axe_mat[axe_mat_row, axe_mat_col].set_xlabel(xlabel = x_label, fontsize = font_size_label)
                    axe_mat[axe_mat_row, axe_mat_col].set_ylabel(ylabel = y_label, fontsize = font_size_label)
                else:
                    continue
    if 'legend' in cure:
        font_size_loge = keywords.get('font_size_loge', 17)
        theta_set_opals = keywords.get('theta_set_opals', None)
        theta_set_picks = keywords.get('theta_set_picks', None)
        theta_set_cocos = keywords.get('theta_set_cocos', None)
        theta_set_pats = keywords.get('theta_set_pats', None)
        coda = keywords.get('coda', False)
        if not coda:
            index = theta_set_picks['post']
            theta_set_opal = theta_set_opals[index].tolist()
            theta_set_coco = theta_set_cocos[slice(0, len(theta_set_cocos)-1)]
            theta_set_pat = theta_set_pats[slice(0, len(theta_set_pats)-1)]
        else:
            index = theta_set_picks['coda']
            theta_set_opal = theta_set_opals[index].tolist() if index != len(theta_set_opals) else mape
            theta_set_coco = theta_set_cocos[slice(index, index+1)]
            theta_set_pat = theta_set_pats[slice(index, index+1)]
        x_limes = (-0.5, len(theta_set_coco)-0.5)
        y_limes = (0, 1)
        for axe_mat_row in range(axe_mat_rows):
            for axe_mat_col in range(axe_mat_cols):
                if axe_mat_col == 0 and axe_mat_row == axe_mat_rows-1:
                    for guide in range(0, len(theta_set_coco)):
                        axe_mat[axe_mat_row, axe_mat_col].axvline(
                            x = guide, ymin = y_limes[0]+(y_limes[1]-y_limes[0])*0.5, ymax = y_limes[1]-(y_limes[1]-y_limes[0])*0.5,
                            color = theta_set_coco[guide], linestyle = '-', label = theta_set_pat[guide], linewidth = 2, marker = '.', markersize = 13
                        )
                    axe_mat[axe_mat_row, axe_mat_col].set_xlim(x_limes[0], x_limes[1])
                    axe_mat[axe_mat_row, axe_mat_col].set_ylim(y_limes[0], y_limes[1])
                    axe_mat[axe_mat_row, axe_mat_col].legend(
                        loc = 'center', fontsize = font_size_loge, frameon = True, framealpha = 0.975, facecolor = 'white',
                        title = ['Parameter Sets', 'Parameter Condition'][coda], title_fontproperties = {'weight': 'book', 'size': font_size_loge}, borderpad = 0.25, handleheight = 1.25
                    )
                else:
                    continue
        if verbose:
            mess = f"{['Post', 'Coda'][coda]}\n\t{theta_set_opal}"
            print(mess)
    return axe_mat

#%%# Retrieve Data [Cellulate /\ Memo]

def _simul_data_load(tasks, path, tag):
    if tasks is None:
        _arcs = os.listdir(path)
        arcs = [arc for arc in _arcs if re.findall(tag+'_'+'(Theta|Trajectory)_Set', arc)]
        _arcs_alp = [arc for arc in arcs if re.findall('_Theta_Set_', arc)]
        _arcs_bet = [arc for arc in arcs if re.findall('_Trajectory_Set_', arc)]
        arcs_alp = [int(re.findall('(\d+)\.pt', arc)[0]) for arc in _arcs_alp]
        arcs_bet = [int(re.findall('(\d+)\.pt', arc)[0]) for arc in _arcs_bet]
        arcs_alp.sort()
        arcs_bet.sort()
        mess = 'Oops! Something went wrong!'
        check = arcs_alp == arcs_bet
        assert check, mess
        _tasks = arcs_alp # _tasks = arcs_bet
        tasks_mini = min(_tasks)
        tasks_maxi = max(_tasks)
    elif type(tasks) is int:
        _tasks = range(tasks)
        tasks_mini = 0
        tasks_maxi = tasks - 1
    else:
        mess = "The format is invalid! The variable 'tasks' must be either 'None' or 'integer'!"
        raise RuntimeError(mess)
    print(f'Load Task Data! {tag}\t{tasks_mini} : {tasks_maxi}\tTotal {len(_tasks)}')
    for task in _tasks:
        if task == tasks_mini:
            label = path+tag+'_Theta_Set_'+str(task)+'.pt'
            theta_set = torch.load(label)
            label = path+tag+'_Trajectory_Set_'+str(task)+'.pt'
            trajectory_set = torch.load(label)
        else:
            label = path+tag+'_Theta_Set_'+str(task)+'.pt'
            _theta_set = torch.load(label)
            theta_set = torch.cat((theta_set, _theta_set), 0)
            label = path+tag+'_Trajectory_Set_'+str(task)+'.pt'
            _trajectory_set = torch.load(label)
            trajectory_set = torch.cat((trajectory_set, _trajectory_set), 0)
        print(f'Load Task Data! {tag}\t{task+1} ~ {len(_tasks)}\tIndex {task}')
    print(f'Load Task Data! {tag}\t{tasks_mini} : {tasks_maxi}\tTotal {len(_tasks)}')
    collect = (theta_set, trajectory_set)
    return collect

def simul_data_load(tasks, path, tag, acts = None, verbose = False):
    if acts is None or len(acts) == 1:
        collect = _simul_data_load(tasks, path, tag)
    else:
        for act in acts:
            act_index = acts.index(act)
            print(f"{'~'*8} Act {act} {'~'*8} {act_index+1} : {len(acts)} {'~'*8} Start {'~'*8}")
            _tag = re.sub('Act_(\d+)_', f'Act_{act}_', tag)
            _path = path
            if verbose: print(f"{' '*8} {_tag}\n{' '*2*8} {_path}")
            if act == 0:
                _tag = re.sub('Observe_(\d+)_', 'Observe_None_', _tag)
                _path = re.sub('Observe_(\d+)/', '', _path)
                if verbose: print(f"{' '*8} {_tag}\n{' '*2*8} {_path}")
            _collect = _simul_data_load(tasks, _path, _tag)
            if act_index == 0:
                collect = _collect
            else:
                theta_set = torch.cat((collect[0], _collect[0]), 0)
                trajectory_set = torch.cat((collect[1], _collect[1]), 0)
                collect = (theta_set, trajectory_set)
            print(f"{'~'*8} Act {act} {'~'*8} {act_index+1} : {len(acts)} {'~'*8} Final {'~'*8}")
    return collect

def retrieve_data_cellulate(memo, cellulate, reiteration, lot, sieve = None, verbose = False):
    if verbose: print(f"{' '*8}{cellulate}")
    prep_data_path = memo['prep_data_path']
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    if lot in [0, 1]:
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{prep_data_path}/{data_path}/Observe_{observe}/' # f'REFERENCE_DATA_BANK/{prep_data_path}/{data_path}/Observe_{observe}/'
        tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    else:
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{prep_data_path}/{data_path}/' # f'REFERENCE_DATA_BANK/{prep_data_path}/{data_path}/'
        tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}_Sieve_{sieve}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate = (theta_set, trajectory_set)
    return data_cellulate

def retrieve_data_memo(memo, cellulates, rules, lot, sieve = None, verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    data_memo = {cellulate: [None]*len(rules) for cellulate in cellulates}
    nook = memo['nook']
    for cellulate in cellulates:
        reiterations = [f'Reiterate_Nook_{cellulate[0]}_{cellulate[1]}', f'Reiterate_{cellulate[0]}_{cellulate[1]}']
        if nook:
            for rule in rules:
                rule_index = rules.index(rule)
                reiteration = reiterations[rule_index]
                data_cellulate = retrieve_data_cellulate(memo, cellulate, reiteration, verbose)
                data_memo[cellulate][rule_index] = data_cellulate[1]
        else:
            reiteration = reiterations[1]
            data_cellulate = retrieve_data_cellulate(memo, cellulate, reiteration, lot, sieve, verbose)
            splitter = int(data_cellulate[1].shape[1]/len(rules))
            rule_split = [(rules.index(rule)*splitter, (rules.index(rule)+1)*splitter) for rule in rules]
            for rule in rules:
                rule_index = rules.index(rule)
                split = rule_split[rule_index]
                data_memo[cellulate][rule_index] = data_cellulate[1][:, split[0]:split[1]]
    return data_memo
