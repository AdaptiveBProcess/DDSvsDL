# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:49:28 2020

@author: Manuel Camargo
"""
import os
import json

import pandas as pd
import configparser as cp

from tensorflow.keras.models import load_model

from support_modules import support as sup
from generators import event_log_predictor as elp


class ModelPredictor():
    """
    This is the man class encharged of the model evaluation
    """

    def __init__(self, parms):
        self.output_route = parms['output_route']
        self.parms = parms
        # load parameters
        self.num_cases = parms['num_cases']
        self.rep = parms['rep']
        self.load_parameters()
        self.model_name, _ = os.path.splitext(parms['model_file'])
        self.model = load_model(os.path.join(parms['folder'],
                                             parms['model_file']))

        self.samples = dict()
        self.predictions = None
        self.run_num = 0

        self.model_def = dict()
        self.read_model_definition(self.parms['model_type'])
        print(self.model_def)
        self.parms['additional_columns'] = self.model_def['additional_columns']
        self.acc = self.execute_predictive_task()

    def execute_predictive_task(self):
        # predict
        self.imp = self.parms['variant']
        self.run_num = 0
        for i in range(0, self.parms['rep']):
            self.predict_values()
            self.run_num += 1
        # export predictions
        self.export_predictions()

    def predict_values(self):
        # Predict values
        executioner = elp.EventLogPredictor()
        results = executioner.predict(self.parms,
                                      self.model,
                                      self.samples,
                                      self.imp,
                                      self.model_def['vectorizer'])
        results = pd.DataFrame(results)
        results['run_num'] = self.run_num
        results['implementation'] = self.imp
        if self.predictions is None:
            self.predictions = results
        else:
            self.predictions = self.predictions.append(results,
                                                       ignore_index=True)

    def load_parameters(self):
        self.parms['num_cases'] = self.num_cases
        # Loading of parameters from training
        path = os.path.join(self.parms['folder'],
                            'parameters',
                            'model_parameters.json')
        with open(path) as file:
            data = json.load(file)
            if 'activity' in data:
                del data['activity']
            self.parms = {**self.parms, **{k: v for k, v in data.items()}}
            self.parms['dim'] = {k: int(v) for k, v in data['dim'].items()}
            if self.parms['one_timestamp']:
                self.parms['scale_args'] = {
                    k: float(v) for k, v in data['scale_args'].items()}
            else:
                for key in data['scale_args'].keys():
                    self.parms['scale_args'][key] = {
                        k: float(v) for k, v in data['scale_args'][key].items()}
            self.parms['index_ac'] = {int(k): v
                                      for k, v in data['index_ac'].items()}
            self.parms['index_rl'] = {int(k): v
                                      for k, v in data['index_rl'].items()}
            file.close()
            self.ac_index = {v: k for k, v in self.parms['index_ac'].items()}
            self.rl_index = {v: k for k, v in self.parms['index_rl'].items()}
        
        self.parms['rep'] = self.rep

    def export_predictions(self):
        if not os.path.exists(self.output_route):
            os.makedirs(self.output_route)
        filename = self.model_name + '.csv'
        self.predictions.to_csv(os.path.join(self.output_route, filename),
                                index=False)

    def read_model_definition(self, model_type):
        Config = cp.ConfigParser(interpolation=None)
        Config.read('models_spec.ini')
        #File name with extension
        self.model_def['additional_columns'] = sup.reduce_list(
            Config.get(model_type,'additional_columns'), dtype='str')
        self.model_def['vectorizer'] = Config.get(model_type, 'vectorizer')

