# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:12:45 2020

@author: Manuel Camargo
"""
import os
import itertools

import pandas as pd
import numpy as np
from operator import itemgetter

from support_modules.readers import log_reader as lr


def split_timeline(log, percentage: float, one_timestamp: bool) -> None:
    """
    Split an event log dataframe to peform split-validation

    Parameters
    ----------
    percentage : float, validation percentage.
    one_timestamp : bool, Support only one timestamp.
    """
    # log = self.log.data.to_dict('records')
    log = sorted(log.data, key=lambda x: x['caseid'])
    for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
        events = list(group)
        events = sorted(events, key=itemgetter('end_timestamp'))
        length = len(events)
        for i in range(0, len(events)):
            events[i]['pos_trace'] = i + 1
            events[i]['trace_len'] = length
    log = pd.DataFrame.from_dict(log)
    log['end_timestamp'] = (
        log['end_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f'))
    if not one_timestamp:
        log['start_timestamp'] = (
            log['start_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f'))
    if 'role' in log.columns:
        log.drop(columns=['role'], inplace=True)
    log.sort_values(by='end_timestamp', ascending=False, inplace=True)

    num_events = int(np.round(len(log)*percentage))

    df_test = log.iloc[:num_events]
    df_train = log.iloc[num_events:]

    # Incomplete final traces
    df_train = df_train.sort_values(by=['caseid', 'pos_trace'], ascending=True)
    inc_traces = pd.DataFrame(df_train.groupby('caseid')
                              .last()
                              .reset_index())
    inc_traces = inc_traces[inc_traces.pos_trace != inc_traces.trace_len]
    inc_traces = inc_traces['caseid'].to_list()
    
    # Drop incomplete traces
    df_test = df_test[~df_test.caseid.isin(inc_traces)]
    df_test = df_test.drop(columns=['trace_len','pos_trace'])

    df_train = df_train[~df_train.caseid.isin(inc_traces)]
    df_train = df_train.drop(columns=['trace_len','pos_trace'])
    
    key = 'end_timestamp' if one_timestamp else 'start_timestamp'
    log_train = (df_train
                 .sort_values(key, ascending=True)
                 .reset_index(drop=True))
    log_test = (df_test
                .sort_values(key, ascending=True)
                .reset_index(drop=True))
    r_column_names = {v: k for k, v in 
                      settings['read_options']['column_names'].items()}
    file = settings['file'].split('.')[0]
    log_train.rename(columns=r_column_names).to_csv(
        os.path.join(settings['output'], file+'_training.csv'), index=False)
    log_test.rename(columns=r_column_names).to_csv(
        os.path.join(settings['output'], file+'_testing.csv'), index=False)
    
# =============================================================================
# Kernel
# =============================================================================


column_names = {'Case ID': 'caseid', 'Activity': 'task',
                'lifecycle:transition': 'event_type', 'Resource': 'user'}

settings = dict()
settings['input'] = 'input_files'
settings['output'] = 'outputs'
settings['file'] = 'inter_PurchasingExample.csv'
settings['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                            'column_names': column_names,
                            'one_timestamp': False,
                            'filter_d_attrib': False,
                            'ns_include': True}

log = lr.LogReader(os.path.join(settings['input'], settings['file']),
                   settings['read_options'])
# Time splitting
split_timeline(log, 0.3, settings['read_options']['one_timestamp'])
