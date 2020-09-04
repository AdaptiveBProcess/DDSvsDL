"""
Created on Fri Jan 10 11:40:46 2020

@author: Manuel Camargo
"""
from sys import stdout
import shutil
import time
import os

import random
import itertools
from operator import itemgetter

import jellyfish as jf
import numpy as np
import pandas as pd
import swifter

from scipy.optimize import linear_sum_assignment

from analyzers import alpha_oracle as ao
from analyzers.alpha_oracle import Rel
from support_modules import slurm_multiprocess as slmp
from support_modules import support as sup

class Evaluator():

    def __init__(self, one_timestamp):
        """constructor"""
        self.one_timestamp = one_timestamp
        self.conn = {'partition': 'main',
                    'mem': str(14000),
                    'env': 'lstm_exp_v3',
                    'script': os.path.join('analyzers', 'slurm_worker.py')}
        self.slurm_workers = 100


    def measure(self, metric, data, feature=None):
        evaluator = self._get_metric_evaluator(metric)
        return evaluator(data, feature)

    def _get_metric_evaluator(self, metric):
        if metric == 'accuracy':
            return self._accuracy_evaluation
        if metric == 'mae_next':
            return self._mae_next_evaluation
        elif metric == 'similarity':
            return self._similarity_evaluation
        elif metric == 'mae_suffix':
            return self._mae_remaining_evaluation
        elif metric == 'els':
            return self._els_metric_evaluation
        elif metric == 'els_min':
            return self._els_min_evaluation
        elif metric == 'mae_log':
            return self._mae_metric_evaluation
        elif metric == 'dl':
            return self._dl_distance_evaluation
        else:
            raise ValueError(metric)

    def _accuracy_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'source']]
        eval_acc = (lambda x:
                    1 if x[feature + '_expect'] == x[feature + '_pred'] else 0)
        data[feature + '_acc'] = data.apply(eval_acc, axis=1)
        # agregate true positives
        data = (data.groupby(['source', 'run_num'])[feature + '_acc']
                .agg(['sum', 'count'])
                .reset_index())
        # calculate accuracy
        data['accuracy'] = np.divide(data['sum'], data['count'])
        return data

    def _mae_next_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'source']]
        ae = (lambda x: np.abs(x[feature + '_expect'] - x[feature + '_pred']))
        data['ae'] = data.apply(ae, axis=1)
        data = (data.groupby(['source', 'run_num'])['ae']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae'}))
        return data

    def _similarity_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'source', 'pref_size']]
        # append all values and create alias
        values = (data[feature + '_pred'].tolist() +
                  data[feature + '_expect'].tolist())
        values = list(set(itertools.chain.from_iterable(values)))
        index = self.create_task_alias(values)
        for col in ['_expect', '_pred']:
            list_to_string = lambda x: ''.join([index[y] for y in x])
            data['suff' + col] = (data[feature + col]
                                  .swifter.progress_bar(False)
                                  .apply(list_to_string))
        # measure similarity between pairs

        def distance(x, y):
            return (1 - (jf.damerau_levenshtein_distance(x, y) /
                         np.max([len(x), len(y)])))
        data['similarity'] = (data[['suff_expect', 'suff_pred']]
                              .swifter.progress_bar(False)
                              .apply(lambda x: distance(x.suff_expect,
                                                        x.suff_pred), axis=1))

        # agregate similarities
        data = (data.groupby(['source', 'run_num', 'pref_size'])['similarity']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'similarity'}))
        data = (pd.pivot_table(data,
                               values='similarity',
                               index=['run_num', 'source'],
                               columns=['pref_size'],
                               aggfunc=np.mean,
                               fill_value=0,
                               margins=True,
                               margins_name='mean')
                .reset_index())
        data = data[data.run_num != 'mean']
        return data

    def _mae_remaining_evaluation(self, data, feature):
        data = data.copy()
        data = data[[(feature + '_expect'), (feature + '_pred'),
                     'run_num', 'source', 'pref_size']]
        ae = (lambda x: np.abs(np.sum(x[feature + '_expect']) -
                               np.sum(x[feature + '_pred'])))
        data['ae'] = data.apply(ae, axis=1)
        data = (data.groupby(['source', 'run_num', 'pref_size'])['ae']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae'}))
        data = (pd.pivot_table(data,
                               values='mae',
                               index=['run_num', 'source'],
                               columns=['pref_size'],
                               aggfunc=np.mean,
                               fill_value=0,
                               margins=True,
                               margins_name='mean')
                .reset_index())
        data = data[data.run_num != 'mean']
        return data

# =============================================================================
# Timed string distance
# =============================================================================

    def _els_metric_evaluation(self, data, feature):
        start = time.time()
        data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.source == 'log']
        alias = self.create_task_alias(data.task.unique())
        alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        num_splits = self.calculate_splits(log_data)
        if num_splits == 1:
            log_data = self.reformat_events(log_data.to_dict('records'),
                                            'task',
                                            alias)
        variants = data[['run_num', 'source']].drop_duplicates()
        variants = variants[variants.source!='log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.source == var['source']) &
                              (data.run_num == var['run_num'])]
            ####
            if num_splits == 1:
                df_matrix, pred_data  = self._calculate_els_cost_matrix(
                    log_data, pred_data, alias, num_splits, alpha_concurrency)
            else:
                log_data = data[data.source == 'log']
                df_matrix, pred_data, log_data = self._calculate_els_cost_matrix(
                log_data, pred_data, alias, num_splits, alpha_concurrency)
            df_matrix = df_matrix.reset_index().set_index(['i','j'])
            cost_matrix = df_matrix[['distance']].unstack().to_numpy()
            # Matching using the hungarian algorithm
            stdout.write("Matching using the hungarian algorithm")
            stdout.flush()
            stdout.write("\n")
            row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                        sim_order=pred_data[idx]['profile'],
                                        log_order=log_data[idy]['profile'],
                                        sim_score=(1-(cost_matrix[idx][idy])),
                                        source=var['source'],
                                        run_num=var['run_num']))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['source', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'els'}))
        end = time.time()
        print('els:', (end - start), sep=' ')
        return data

    def _calculate_els_cost_matrix(self, log_data, pred_data, alias, num_splits, alpha_concurrency=None):
        if num_splits == 1:
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                              'task',
                                              alias)
            mx_len = len(log_data)
            # Create cost matrix
            df_matrix = list()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    df_matrix.append({
                        's_1': pred_data[i]['profile'],
                        's_2': log_data[j]['profile'],
                        'p_1': pred_data[i]['dur_act_norm'],
                        'p_2': log_data[j]['dur_act_norm'],
                        'w_1': pred_data[i]['wait_act_norm'],
                        'w_2': log_data[j]['wait_act_norm'],
                        'length': max(len(pred_data[i]['profile']),
                                      len(log_data[j]['profile'])),
                        'i': i, 'j': j})
            df_matrix = pd.DataFrame(df_matrix)
            df_matrix['distance'] = df_matrix.apply(
                lambda x: self.tsd_alpha(
                    x.s_1, x.s_2, x.p_1, x.p_2, x.w_1, x.w_2,
                                        alpha_concurrency.oracle), axis=1)
            df_matrix['distance'] = df_matrix['distance']/df_matrix['length']
            return df_matrix, pred_data
        else:
            folder = sup.folder_id()
            output = os.path.join('analyzers', folder)
            if not os.path.exists(output):
                os.makedirs(output)
            log_cases = log_data.caseid.unique()
            pred_cases = pred_data.caseid.unique()
            ranges = self.define_ranges(log_cases, num_splits)
            ranges = list(itertools.product(*[ranges, ranges]))
            pred_cases = pd.DataFrame(pred_cases, columns=['caseid'])
            pred_cases['i'] = pred_cases.index
            log_cases = pd.DataFrame(log_cases, columns=['caseid'])
            log_cases['j'] = log_cases.index
            oracle = [{'key': [ord(c) for c in k], 'value':v.value}
                for k,v in alpha_concurrency.oracle.items()]
            oracle_path = os.path.join(output, 'oracle.csv')
            pd.DataFrame(oracle).to_csv(oracle_path, index=False)
            pred_cases.to_csv(os.path.join(output, 'pred_index.csv'), index=False)
            log_cases.to_csv(os.path.join(output, 'log_index.csv'), index=False)
            pred_data.to_csv(os.path.join(output, 'pred_data.csv'), index=False)
            log_data.to_csv(os.path.join(output, 'log_data.csv'), index=False)
            sup.create_json(alias,os.path.join(output, 'alias.json'))
            args = [
                {'t': r[0]['min'], 'd': r[0]['max'],
                 'l': r[1]['min'], 'r': r[1]['max'],
                 'a': oracle_path, 'm': 'els'} for r in ranges]
            stdout.write("Calculating values ")
            stdout.flush()
            stdout.write("\n")
            mprocessor = slmp.HPC_Multiprocess(self.conn,
                                               args,
                                               output,
                                               output,
                                               self.slurm_workers,
                                               timeout=5)
            mprocessor.parallelize()
            stdout.write("Joining matrix ")
            stdout.flush()
            stdout.write("\n")
            df_matrix = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'out_')]
            df_matrix = pd.concat(df_matrix, axis=0, ignore_index=True)
            df_matrix.sort_values(by=['i', 'j'], inplace=True)
            log_data = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'log_data_')]
            log_data = pd.concat(log_data, axis=0, ignore_index=True)
            log_data.sort_values(by='idx', inplace=True)
            log_data = log_data.to_dict('records')
            pred_data = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'pred_data_')]
            pred_data = pd.concat(pred_data, axis=0, ignore_index=True)
            pred_data.sort_values(by='idx', inplace=True)
            pred_data = pred_data.to_dict('records')
            shutil.rmtree(output)
            if os.path.exists(output):
                os.rmdir(output)
            return df_matrix, pred_data, log_data

    def _els_min_evaluation(self, data, feature):
        start = time.time()
        data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.source == 'log']
        alias = self.create_task_alias(data.task.unique())
        alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
        # log reformating
        log_data = self.reformat_events(log_data.to_dict('records'),
                                        'task',
                                        alias)
        variants = data[['run_num', 'source']].drop_duplicates()
        variants = variants[variants.source!='log'].to_dict('records')
        similarity = list()
        for var in variants:
            pred_data = data[(data.source == var['source']) &
                              (data.run_num == var['run_num'])]
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                              'task',
                                              alias)
            temp_log_data = log_data.copy()
            for i in range(0, len(pred_data)):
                min_dist = sim = self.tsd_alpha(
                                        pred_data[i]['profile'],
                                        temp_log_data[0]['profile'],
                                        pred_data[i]['dur_act_norm'],
                                        temp_log_data[0]['dur_act_norm'],
                                        pred_data[i]['wait_act_norm'],
                                        temp_log_data[0]['wait_act_norm'],
                                        alpha_concurrency.oracle)

                min_idx = 0
                for j in range(1, len(temp_log_data)):
                    sim = self.tsd_alpha(
                                        pred_data[i]['profile'],
                                        temp_log_data[j]['profile'],
                                        pred_data[i]['dur_act_norm'],
                                        temp_log_data[j]['dur_act_norm'],
                                        pred_data[i]['wait_act_norm'],
                                        temp_log_data[j]['wait_act_norm'],
                                        alpha_concurrency.oracle)
                    if min_dist > sim:
                        min_dist = sim
                        min_idx = j
                length = np.max([len(pred_data[i]['profile']),
                                  len(temp_log_data[min_idx]['profile'])])
                similarity.append(dict(caseid=pred_data[i]['caseid'],
                                        sim_order=pred_data[i]['profile'],
                                        log_order=temp_log_data[min_idx]['profile'],
                                        sim_score=(1-(min_dist/length)),
                                        source=var['source'],
                                        run_num = var['run_num']))
                del temp_log_data[min_idx]
        data = pd.DataFrame(similarity)
        data = (data.groupby(['source', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'els'}))
        end = time.time()
        print('els min:', (end - start), sep=' ')
        return data

    def tsd_alpha(self, s_1, s_2, p_1, p_2, w_1, w_2, alpha_concurrency):
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

# =============================================================================
# dl distance
# =============================================================================

    def _dl_distance_evaluation(self, data, feature):
        data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.source == 'log']
        alias = self.create_task_alias(data.task.unique())
        # log reformating
        num_splits = self.calculate_splits(log_data)
        if num_splits == 1:
            log_data = self.reformat_events(log_data.to_dict('records'),
                                            'task',
                                            alias)
        variants = data[['run_num', 'source']].drop_duplicates()
        variants = variants[variants.source!='log'].to_dict('records')
        similarity = list()
        # ts = time.time()
        for var in variants:
            pred_data = data[(data.source == var['source']) &
                              (data.run_num == var['run_num'])]
            ####
            if num_splits == 1:
                df_matrix, pred_data  = self._calculate_dl_cost_matrix(
                    log_data, pred_data, alias, num_splits)
            else:
                log_data = data[data.source == 'log']
                df_matrix, pred_data, log_data = self._calculate_dl_cost_matrix(
                log_data, pred_data, alias, num_splits)
            df_matrix = df_matrix.reset_index().set_index(['i','j'])
            cost_matrix = df_matrix[['distance']].unstack().to_numpy()
            # Matching using the hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                        sim_order=pred_data[idx]['profile'],
                                        log_order=log_data[idy]['profile'],
                                        sim_score=(1-(cost_matrix[idx][idy])),
                                        source=var['source'],
                                        run_num=var['run_num']))
        # te = time.time()
        # print((te - ts))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['source', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'dl'}))
        return data

    def _calculate_dl_cost_matrix(self, log_data, pred_data, alias, num_splits):
        if num_splits == 1:
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                              'task',
                                              alias)
            mx_len = len(log_data)
            # Create cost matrix
            df_matrix = list()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    df_matrix.append({
                        's_1': pred_data[i]['profile'],
                        's_2': log_data[j]['profile'],
                        'length': max(len(pred_data[i]['profile']),
                                      len(log_data[j]['profile'])),
                        'i': i, 'j': j})
            df_matrix = pd.DataFrame(df_matrix)
            df_matrix['distance'] = df_matrix.apply(
                lambda x: self.calculate_distances(x.s_1, x.s_2), axis=1)
            df_matrix['distance'] = df_matrix['distance']/df_matrix['length']
            return df_matrix, pred_data
        else:
            folder = sup.folder_id()
            output = os.path.join('analyzers', folder)
            if not os.path.exists(output):
                os.makedirs(output)
            log_cases = log_data.caseid.unique()
            pred_cases = pred_data.caseid.unique()
            ranges = self.define_ranges(log_cases, num_splits)
            ranges = list(itertools.product(*[ranges, ranges]))
            pred_cases = pd.DataFrame(pred_cases, columns=['caseid'])
            pred_cases['i'] = pred_cases.index
            log_cases = pd.DataFrame(log_cases, columns=['caseid'])
            log_cases['j'] = log_cases.index
            pred_cases.to_csv(os.path.join(output, 'pred_index.csv'), index=False)
            log_cases.to_csv(os.path.join(output, 'log_index.csv'), index=False)
            pred_data.to_csv(os.path.join(output, 'pred_data.csv'), index=False)
            log_data.to_csv(os.path.join(output, 'log_data.csv'), index=False)
            sup.create_json(alias,os.path.join(output, 'alias.json'))
            args = [
                {'t': r[0]['min'], 'd': r[0]['max'],
                 'l': r[1]['min'], 'r': r[1]['max'],
                 'm': 'dl'} for r in ranges]
            stdout.write("Calculating values ")
            stdout.flush()
            stdout.write("\n")
            mprocessor = slmp.HPC_Multiprocess(self.conn,
                                               args,
                                               output,
                                               output,
                                               self.slurm_workers,
                                               timeout=5)
            mprocessor.parallelize()
            stdout.write("Joining matrix ")
            stdout.flush()
            stdout.write("\n")
            df_matrix = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'out_')]
            df_matrix = pd.concat(df_matrix, axis=0, ignore_index=True)
            df_matrix.sort_values(by=['i', 'j'], inplace=True)
            log_data = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'log_data_')]
            log_data = pd.concat(log_data, axis=0, ignore_index=True)
            log_data.sort_values(by='idx', inplace=True)
            log_data = log_data.to_dict('records')
            pred_data = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'pred_data_')]
            pred_data = pd.concat(pred_data, axis=0, ignore_index=True)
            pred_data.sort_values(by='idx', inplace=True)
            pred_data = pred_data.to_dict('records')
            shutil.rmtree(output)
            if os.path.exists(output):
                os.rmdir(output)
            return df_matrix, pred_data, log_data

    @staticmethod
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

    # @staticmethod
    # def calculate_distances(serie1, serie2, id1, id2):
    #     """
    #     Parameters
    #     ----------
    #     serie1 : list
    #     serie2 : list
    #     id1 : index of the list 1
    #     id2 : index of the list 2

    #     Returns
    #     -------
    #     dl : float value
    #     ae : absolute error value
    #     """
    #     length = np.max([len(serie1[id1]['profile']),
    #                      len(serie2[id2]['profile'])])
    #     d_l = jf.damerau_levenshtein_distance(
    #         ''.join(serie1[id1]['profile']),
    #         ''.join(serie2[id2]['profile']))/length
    #     return d_l

# =============================================================================
# mae distance
# =============================================================================

    def _mae_metric_evaluation(self, data, feature):
        data = self.add_calculated_times(data)
        data = self.scaling_data(data)
        log_data = data[data.source == 'log']
        alias = self.create_task_alias(data.task.unique())
        # log reformating
        num_splits = self.calculate_splits(log_data)
        if num_splits == 1:
            log_data = self.reformat_events(log_data.to_dict('records'),
                                            'task',
                                            alias)
        variants = data[['run_num', 'source']].drop_duplicates()
        variants = variants[variants.source!='log'].to_dict('records')
        similarity = list()
        # ts = time.time()
        for var in variants:
            pred_data = data[(data.source == var['source']) &
                              (data.run_num == var['run_num'])]
            ####
            if num_splits == 1:
                df_matrix, pred_data  = self._calculate_mae_cost_matrix(
                    log_data, pred_data, alias, num_splits)
            else:
                log_data = data[data.source == 'log']
                df_matrix, pred_data, log_data = self._calculate_mae_cost_matrix(
                log_data, pred_data, alias, num_splits)
            df_matrix = df_matrix.reset_index().set_index(['i','j'])
            cost_matrix = df_matrix[['distance']].unstack().to_numpy()
            # Matching using the hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
            # Create response
            for idx, idy in zip(row_ind, col_ind):
                similarity.append(dict(caseid=pred_data[idx]['caseid'],
                                        sim_order=pred_data[idx]['profile'],
                                        log_order=log_data[idy]['profile'],
                                        sim_score=cost_matrix[idx][idy],
                                        source=var['source'],
                                        run_num=var['run_num']))
        # te = time.time()
        # print((te - ts))
        data = pd.DataFrame(similarity)
        data = (data.groupby(['source', 'run_num'])['sim_score']
                .agg(['mean'])
                .reset_index()
                .rename(columns={'mean': 'mae_log'}))
        return data

    # def _mae_metric_evaluation(self, data, feature):
    #     """
    #     mae distance between logs

    #     Parameters
    #     ----------
    #     log_data : list of events
    #     simulation_data : list simulation event log

    #     Returns
    #     -------
    #     similarity : float

    #     """
    #     data = self.add_calculated_times(data)
    #     data = self.scaling_data(data)
    #     log_data = data[data.source == 'log']
    #     alias = self.create_task_alias(data.task.unique())
    #     # alpha_concurrency = ao.AlphaOracle(log_data, alias, True, True)
    #     # log reformating
    #     log_data = self.reformat_events(log_data.to_dict('records'),
    #                                     'task',
    #                                     alias)
    #     variants = data[['run_num', 'source']].drop_duplicates()
    #     variants = variants[variants.source != 'log'].to_dict('records')
    #     similarity = list()
    #     for var in variants:
    #         pred_data = data[(data.source == var['source']) &
    #                          (data.run_num == var['run_num'])]
    #         pred_data = self.reformat_events(pred_data.to_dict('records'),
    #                                          'task',
    #                                          alias)
    #         mx_len = len(log_data)
    #         ae_matrix = [[0 for c in range(mx_len)] for r in range(mx_len)]
    #         # Create cost matrix
    #         # start = timer()
    #         for i in range(0, mx_len):
    #             for j in range(0, mx_len):
    #                 cicle_time_s1 = (pred_data[i]['end_time'] -
    #                                  pred_data[i]['start_time']).total_seconds()
    #                 cicle_time_s2 = (log_data[j]['end_time'] -
    #                                  log_data[j]['start_time']).total_seconds()
    #                 ae = np.abs(cicle_time_s1 - cicle_time_s2)
    #                 ae_matrix[i][j] = ae
    #         # end = timer()
    #         # print(end - start)
    #         ae_matrix = np.array(ae_matrix)
    #         # Matching using the hungarian algorithm
    #         row_ind, col_ind = linear_sum_assignment(np.array(ae_matrix))
    #         # Create response
    #         for idx, idy in zip(row_ind, col_ind):
    #             similarity.append(dict(caseid=pred_data[idx]['caseid'],
    #                                    sim_order=pred_data[idx]['profile'],
    #                                    log_order=log_data[idy]['profile'],
    #                                    sim_score=(ae_matrix[idx][idy]),
    #                                    source=var['source'],
    #                                    run_num=var['run_num']))
    #     data = pd.DataFrame(similarity)
    #     data = (data.groupby(['source', 'run_num'])['sim_score']
    #             .agg(['mean'])
    #             .reset_index()
    #             .rename(columns={'mean': 'mae_log'}))
    #     return data

    def _calculate_mae_cost_matrix(self, log_data, pred_data, alias, num_splits):
        if num_splits == 1:
            pred_data = self.reformat_events(pred_data.to_dict('records'),
                                              'task',
                                              alias)
            mx_len = len(log_data)
            # Create cost matrix
            df_matrix = list()
            for i in range(0, mx_len):
                for j in range(0, mx_len):
                    df_matrix.append({
                        'cicle_time_s1': (pred_data[i]['end_time'] -
                                          pred_data[i]['start_time']).total_seconds(),
                        'cicle_time_s2': (log_data[j]['end_time'] -
                                          log_data[j]['start_time']).total_seconds(),
                        'i': i, 'j': j})
            df_matrix = pd.DataFrame(df_matrix)
            df_matrix['distance'] = np.abs(df_matrix.cicle_time_s1 - df_matrix.cicle_time_s2)
            return df_matrix, pred_data
        else:
            folder = sup.folder_id()
            output = os.path.join('analyzers', folder)
            if not os.path.exists(output):
                os.makedirs(output)
            log_cases = log_data.caseid.unique()
            pred_cases = pred_data.caseid.unique()
            ranges = self.define_ranges(log_cases, num_splits)
            ranges = list(itertools.product(*[ranges, ranges]))
            pred_cases = pd.DataFrame(pred_cases, columns=['caseid'])
            pred_cases['i'] = pred_cases.index
            log_cases = pd.DataFrame(log_cases, columns=['caseid'])
            log_cases['j'] = log_cases.index
            pred_cases.to_csv(os.path.join(output, 'pred_index.csv'), index=False)
            log_cases.to_csv(os.path.join(output, 'log_index.csv'), index=False)
            pred_data.to_csv(os.path.join(output, 'pred_data.csv'), index=False)
            log_data.to_csv(os.path.join(output, 'log_data.csv'), index=False)
            sup.create_json(alias,os.path.join(output, 'alias.json'))
            args = [
                {'t': r[0]['min'], 'd': r[0]['max'],
                 'l': r[1]['min'], 'r': r[1]['max'],
                 'm': 'mae_log'} for r in ranges]
            stdout.write("Calculating values ")
            stdout.flush()
            stdout.write("\n")
            mprocessor = slmp.HPC_Multiprocess(self.conn,
                                               args,
                                               output,
                                               output,
                                               self.slurm_workers,
                                               timeout=5)
            mprocessor.parallelize()
            stdout.write("Joining matrix ")
            stdout.flush()
            stdout.write("\n")
            df_matrix = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'out_')]
            df_matrix = pd.concat(df_matrix, axis=0, ignore_index=True)
            df_matrix.sort_values(by=['i', 'j'], inplace=True)
            log_data = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'log_data_')]
            log_data = pd.concat(log_data, axis=0, ignore_index=True)
            log_data.sort_values(by='idx', inplace=True)
            log_data = log_data.to_dict('records')
            pred_data = [pd.read_csv(os.path.join(output, f))
                      for f in self.create_file_list(output, 'pred_data_')]
            pred_data = pd.concat(pred_data, axis=0, ignore_index=True)
            pred_data.sort_values(by='idx', inplace=True)
            pred_data = pred_data.to_dict('records')
            shutil.rmtree(output)
            if os.path.exists(output):
                os.rmdir(output)
            return df_matrix, pred_data, log_data


# =============================================================================
# Support methods
# =============================================================================
    @staticmethod
    def calculate_splits(df, max_cases=1000):
        print(len(df.caseid.unique()))
        # calculate the number of bytes a row occupies
        n_splits = int(np.ceil(len(df.caseid.unique()) / max_cases))
        return n_splits

    @staticmethod
    def folding_creation(df, splits, output):
        idxs = [x for x in range(0, len(df), round(len(df)/splits))]
        idxs.append(len(df))
        folds = [pd.DataFrame(df.iloc[idxs[i-1]:idxs[i]])
                 for i in range(1, len(idxs))]
        # Export folds
        file_names = list()
        for i, fold in enumerate(folds):
            file_name = os.path.join(output,'split_'+str(i+1)+'.csv')
            fold.to_csv(file_name, index=False)
            file_names.append(file_name)
        return file_names

    @staticmethod
    def define_ranges(data, num_folds):
        num_events = int(np.round(len(data)/num_folds))
        folds = list()
        for i in range(0, num_folds):
            sidx = i * num_events
            eidx = (i + 1) * num_events
            if i == 0:
                folds.append({'min': 0, 'max': eidx})
            elif i == (num_folds - 1):
                folds.append({'min': sidx, 'max': len(data)})
            else:
                folds.append({'min': sidx, 'max': eidx})
        return folds

    @staticmethod
    def create_task_alias(categories):
        """
        Create string alias for tasks names or tuples of tasks-roles names

        Parameters
        ----------
        features : list

        Returns
        -------
        alias : alias dictionary

        """
        variables = sorted(categories)
        characters = [chr(i) for i in range(0, len(variables))]
        aliases = random.sample(characters, len(variables))
        alias = dict()
        for i, _ in enumerate(variables):
            alias[variables[i]] = aliases[i]
        return alias

    def add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['duration'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            ordk = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            events = sorted(events, key=itemgetter(ordk))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instant since there is no previous timestamp
                if self.one_timestamp:
                    if i == 0:
                        dur = 0
                    else:
                        dur = (events[i]['end_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                else:
                    dur = (events[i]['end_timestamp'] -
                           events[i]['start_timestamp']).total_seconds()
                    if i == 0:
                        wit = 0
                    else:
                        wit = (events[i]['start_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                    events[i]['waiting'] = wit
                events[i]['duration'] = dur
        return pd.DataFrame.from_dict(log)

    def scaling_data(self, data):
        """
        Scales times values activity based

        Parameters
        ----------
        data : dataframe

        Returns
        -------
        data : dataframe with normalized times

        """
        df_modif = data.copy()
        np.seterr(divide='ignore')
        summ = data.groupby(['task'])['duration'].max().to_dict()
        dur_act_norm = (lambda x: x['duration']/summ[x['task']]
                        if summ[x['task']] > 0 else 0)
        df_modif['dur_act_norm'] = df_modif.apply(dur_act_norm, axis=1)
        if not self.one_timestamp:
            summ = data.groupby(['task'])['waiting'].max().to_dict()
            wait_act_norm = (lambda x: x['waiting']/summ[x['task']]
                            if summ[x['task']] > 0 else 0)
            df_modif['wait_act_norm'] = df_modif.apply(wait_act_norm, axis=1)
        return df_modif

    def reformat_events(self, data, features, alias):
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
        if self.one_timestamp:
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
                            'end_time': trace[-1][sort_key]},
                         **temp_dict}
            temp_data.append(temp_dict)
        return sorted(temp_data, key=itemgetter('start_time'))

    @staticmethod
    def create_file_list(path, prefix):
        file_list = list()
        for root, dirs, files in os.walk(path):
            for f in files:
                if prefix in f:
                    file_list.append(f)
        return file_list

# #%% kernel
# if __name__ == '__main__':
#     # freeze_support()
#     results = pd.read_csv('C:/Users/Manuel Camargo/Documents/Repositorio/experiments/sc_evaluator/results.csv')
#     results['start_timestamp'] =  pd.to_datetime(results['start_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
#     results['end_timestamp'] =  pd.to_datetime(results['end_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')
#     results = results[results.run_num==0]

#     evaluator = Evaluator(False)
#     els = evaluator.measure('els', results)
#     print(els)
