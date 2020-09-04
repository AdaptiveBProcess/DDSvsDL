# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:42:44 2020

@author: Manuel Camargo
"""
import os
import sys
import getopt
import json
import numpy as np
import pandas as pd
from alpha_oracle import Rel
import jellyfish as jf
import itertools
from operator import itemgetter


def main(argv):
    def catch_parameter(opt):
        """Change the captured parameters names"""
        switch = {'-t': 'i_min', '-d': 'i_max',
                  '-l': 'j_min','-r': 'j_max',
                  '-a': 'oracle', '-o': 'output', '-m': 'metric'}
        try:
            return switch[opt]
        except:
            raise Exception('Invalid option ' + opt)
    # Catch parameters by console
    parameters = dict()
    try:
        opts, _ = getopt.getopt(
            argv, "hf:a:o:m:t:d:l:r:", ['i_min=', 'i_max=',
                                        'j_min=', 'j_max=',
                                        'oracle=', 'output=', 'metric='])
        for opt, arg in opts:
            key = catch_parameter(opt)
            if key in ['i_min', 'i_max', 'j_min', 'j_max']:
                parameters[key] = int(arg)
            else:
                parameters[key] = arg
    except getopt.GetoptError:
        print('Invalid option')
        sys.exit(2)

    print(parameters)
    if parameters['metric'] == 'els':
        pred_data, log_data, alias, oracle, i_index, j_index = read_inputs(parameters)
        log_data = reformat_events(log_data.to_dict('records'),
                                   'task', alias, False, j_index)
        pred_data = reformat_events(pred_data.to_dict('records'),
                                    'task', alias, False, i_index)
        log_mx_len = len(log_data)
        pred_mx_len = len(pred_data)
        # Create cost matrix
        df_matrix = list()
        for i in range(0, pred_mx_len):
            for j in range(0, log_mx_len):
                df_matrix.append({
                    's_1': pred_data[i]['profile'],
                    's_2': log_data[j]['profile'],
                    'p_1': pred_data[i]['dur_act_norm'],
                    'p_2': log_data[j]['dur_act_norm'],
                    'w_1': pred_data[i]['wait_act_norm'],
                    'w_2': log_data[j]['wait_act_norm'],
                    'length': max(len(pred_data[i]['profile']),
                                  len(log_data[j]['profile'])),
                    'i': pred_data[i]['idx'],
                    'j': log_data[j]['idx']})
        df_matrix = pd.DataFrame(df_matrix)
        df_matrix['distance'] = df_matrix.apply(
            lambda x: tsd_alpha(
                x.s_1, x.s_2, x.p_1, x.p_2, x.w_1, x.w_2,
                                    oracle), axis=1)
        df_matrix['distance'] = df_matrix['distance']/df_matrix['length']
    elif parameters['metric'] == 'dl':
        pred_data, log_data, alias, i_index, j_index = read_inputs(parameters)
        log_data = reformat_events(log_data.to_dict('records'),
                                   'task', alias, False, j_index)
        pred_data = reformat_events(pred_data.to_dict('records'),
                                    'task', alias, False, i_index)
        log_mx_len = len(log_data)
        pred_mx_len = len(pred_data)
        # Create cost matrix
        df_matrix = list()
        for i in range(0, pred_mx_len):
            for j in range(0, log_mx_len):
                df_matrix.append({
                    's_1': pred_data[i]['profile'],
                    's_2': log_data[j]['profile'],
                    'length': max(len(pred_data[i]['profile']),
                                  len(log_data[j]['profile'])),
                    'i': pred_data[i]['idx'],
                    'j': log_data[j]['idx']})
        df_matrix = pd.DataFrame(df_matrix)
        df_matrix['distance'] = df_matrix.apply(
            lambda x: calculate_distances(x.s_1, x.s_2), axis=1)
        df_matrix['distance'] = df_matrix['distance']/df_matrix['length']
    elif parameters['metric'] == 'mae_log':
        pred_data, log_data, alias, i_index, j_index = read_inputs(parameters)
        log_data = reformat_events(log_data.to_dict('records'),
                                   'task', alias, False, j_index)
        pred_data = reformat_events(pred_data.to_dict('records'),
                                    'task', alias, False, i_index)
        log_mx_len = len(log_data)
        pred_mx_len = len(pred_data)
        # Create cost matrix
        df_matrix = list()
        for i in range(0, pred_mx_len):
            for j in range(0, log_mx_len):
                df_matrix.append({
                'cicle_time_s1': (pred_data[i]['end_time'] -
                                  pred_data[i]['start_time']).total_seconds(),
                'cicle_time_s2': (log_data[j]['end_time'] -
                                  log_data[j]['start_time']).total_seconds(),
                'i': pred_data[i]['idx'],
                'j': log_data[j]['idx']})
        df_matrix = pd.DataFrame(df_matrix)
        df_matrix['distance'] = np.abs(df_matrix.cicle_time_s1 - df_matrix.cicle_time_s2)
    coord= ('i_'+str(parameters['i_min'])+'_'+str(parameters['i_max'])
            +'_j_'+str(parameters['j_min'])+'_'+str(parameters['j_max']))
    i_coord = ('i_'+str(parameters['i_min'])+'_'+str(parameters['i_max']))
    j_coord = ('j_'+str(parameters['j_min'])+'_'+str(parameters['j_max']))
    df_matrix.to_csv(os.path.join(
        parameters['output'],'out_'+coord+'.csv'), index=False)
    if not os.path.exists(os.path.join(parameters['output'],
                                       'pred_data_'+i_coord+'.csv')):
        pd.DataFrame(pred_data).to_csv(os.path.join(
            parameters['output'], 'pred_data_'+i_coord+'.csv'), index=False)
    if not os.path.exists(os.path.join(parameters['output'],
                                       'log_data_'+j_coord+'.csv')):
        pd.DataFrame(log_data).to_csv(os.path.join(
            parameters['output'], 'log_data_'+j_coord+'.csv'), index=False)
    print('COMPLETED')

def read_inputs(parameters):
    output = parameters['output']
    pred_cases = pd.read_csv(os.path.join(output, 'pred_index.csv'))
    pred_cases = pred_cases[(pred_cases.i>=parameters['i_min']) &
                            (pred_cases.i<parameters['i_max'])]
    i_index = {record['caseid']: record['i'] for record in pred_cases.to_dict('records')}
    pred_cases = pred_cases['caseid'].to_list()
    pred_data = pd.read_csv(os.path.join(output, 'pred_data.csv'))
    pred_data = pred_data[pred_data.caseid.isin(pred_cases)]
    pred_data['start_timestamp'] =  pd.to_datetime(
        pred_data['start_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
    pred_data['end_timestamp'] =  pd.to_datetime(
        pred_data['end_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
    log_cases = pd.read_csv(os.path.join(output, 'log_index.csv'))
    log_cases = log_cases[(log_cases.j>=parameters['j_min']) &
                          (log_cases.j<parameters['j_max'])]
    j_index = {record['caseid']: record['j'] for record in log_cases.to_dict('records')}
    log_cases = log_cases['caseid'].to_list()
    log_data = pd.read_csv(os.path.join(output, 'log_data.csv'))
    log_data = log_data[log_data.caseid.isin(log_cases)]
    log_data['start_timestamp'] =  pd.to_datetime(
        log_data['start_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
    log_data['end_timestamp'] =  pd.to_datetime(
        log_data['end_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
    with open(os.path.join(output, 'alias.json')) as file:
        data = json.load(file)
        alias = {k: v for k, v in data.items()}
    if parameters['metric'] == 'els':
        oracle = pd.read_csv(parameters['oracle'])
        oracle['key'] = oracle.key.apply(string_list, args=('chr',))
        oracle = oracle.to_dict('records')
        replace_value = {1: Rel.FOLLOWS,
                         2: Rel.PRECEDES,
                         3: Rel.NOT_CONNECTED,
                         4: Rel.PARALLEL}
        oracle = {
            (x['key'][0], x['key'][1]): replace_value[x['value']] for x in oracle}
        return pred_data, log_data, alias, oracle, i_index, j_index
    else:
        return pred_data, log_data, alias, i_index, j_index

def tsd_alpha(s_1, s_2, p_1, p_2, w_1, w_2, alpha_concurrency):
    """
    Compute the Damerau-Levenshtein distance between two given
    strings (s_1 and s_2)
    Parameters
    ----------
    comp_sec : dict
    alpha_concurrency : dict
    Returns
    -------
    Float
    """

    def calculate_cost(s1_idx, s2_idx):
        t_1 = p_1[s1_idx] + w_1[s1_idx]
        if t_1 > 0:
            b_1 = (p_1[s1_idx]/t_1)
            cost = ((b_1*np.abs(p_2[s2_idx]-p_1[s1_idx])) +
                    ((1 - b_1)*np.abs(w_2[s2_idx]-w_1[s1_idx])))
        else:
            cost = 0
        return cost

    dist = {}
    lenstr1 = len(s_1)
    lenstr2 = len(s_2)
    for i in range(-1, lenstr1+1):
        dist[(i, -1)] = i+1
    for j in range(-1, lenstr2+1):
        dist[(-1, j)] = j+1
    for i in range(0, lenstr1):
        for j in range(0, lenstr2):
            if s_1[i] == s_2[j]:
                cost = calculate_cost(i, j)
            else:
                cost = 1
            dist[(i, j)] = min(
                dist[(i-1, j)] + 1, # deletion
                dist[(i, j-1)] + 1, # insertion
                dist[(i-1, j-1)] + cost # substitution
                )
            if i and j and s_1[i] == s_2[j-1] and s_1[i-1] == s_2[j]:
                if alpha_concurrency[(s_1[i], s_2[j])] == Rel.PARALLEL:
                    cost = calculate_cost(i, j-1)
                dist[(i, j)] = min(dist[(i, j)], dist[i-2, j-2] + cost)  # transposition
    return dist[lenstr1-1, lenstr2-1]


def calculate_distances(serie1, serie2):
    """
    Parameters
    ----------
    serie1 : list
    serie2 : list
    Returns
    -------
    dl : float value
    ae : absolute error value
    """
    d_l = jf.damerau_levenshtein_distance(
        ''.join(serie1),
        ''.join(serie2))
    return d_l

def string_list(input, dtype='int'):
    text = str(input).replace('[', '').replace(']', '')
    text = [x for x in text.split(',') if x != ' ']
    # text = re.sub(' +', ' ', text)
    # text = text.strip()
    for number in text:
        if dtype=='int':
            return list([int(x) for x in text])
        elif dtype=='float':
            return list([float(x) for x in text])
        elif dtype=='str':
            return list([x.strip() for x in text])
        elif dtype=='chr':
            return list([chr(int(x)) for x in text])
        else:
            raise ValueError(dtype)

def reformat_events(data, features, alias, one_timestamp, index):
    """Creates series of activities, roles and relative times per trace.
    parms:
        log_df: dataframe.
        ac_table (dict): index of activities.
        rl_table (dict): index of roles.
    Returns:
        list: lists of activities, roles and relative times.
    """
    # Update alias
    if isinstance(features, list):
        [x.update(dict(alias=alias[(x[features[0]],
                                         x[features[1]])])) for x in data]
    else:
        [x.update(dict(alias=alias[x[features]])) for x in data]
    temp_data = list()
    # define ordering keys and columns
    if one_timestamp:
        columns = ['alias', 'duration', 'dur_act_norm']
        sort_key = 'end_timestamp'
    else:
        sort_key = 'start_timestamp'
        columns = ['alias', 'duration',
                   'dur_act_norm', 'waiting', 'wait_act_norm']
    data = sorted(data, key=lambda x: (x['caseid'], x[sort_key]))
    for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
        trace = list(group)
        temp_dict = dict()
        for col in columns:
            serie = [y[col] for y in trace]
            if col == 'alias':
                temp_dict = {**{'profile': serie}, **temp_dict}
            else:
                serie = [y[col] for y in trace]
            temp_dict = {**{col: serie}, **temp_dict}
        temp_dict = {**{'caseid': key, 'start_time': trace[0][sort_key],
                        'end_time': trace[-1][sort_key], 'idx': index[key]},
                     **temp_dict}
        temp_data.append(temp_dict)
    return sorted(temp_data, key=itemgetter('start_time'))

if __name__ == '__main__':
    print(os.getcwd())
    main(sys.argv[1:])
