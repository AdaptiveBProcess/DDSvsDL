# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:56:25 2019

@author: Manuel Camargo
"""
import os
import subprocess
from xml.dom import minidom

import pandas as pd
from pandas.api.types import is_numeric_dtype

from support_modules import support as sup
from support_modules.support import timeit
from support_modules.readers import log_reader as lr
from generators import model_predictor as mp
from analyzers import sim_evaluator as ev


class Evaluator():
    """
    Main class of the Simulation Models Discoverer
    """

    def __init__(self, settings):
        """constructor"""
        self.settings = settings
        self.sim_values = list()
        

    def execute_pipeline(self) -> None:
        exec_times = dict()
        self.create_output_folders()
        self.read_inputs(log_time=exec_times)
        models = self.find_models(log_time=exec_times)
        results = list()
        for model in models:
            results.append(
                self.generate_log(os.path.splitext(model)[1], model))
        results = pd.concat(results, axis=0, ignore_index=True)
        results = results.append(self.log_test, ignore_index=True)
        self.evaluate_predict_log(results, self.settings, log_time=exec_times)
        self.save_times(exec_times, self.settings)
        print("-- End of trial --")
        
    def read_and_evaluate(self) -> None:
        print('-- Reading and evaluating --')
        exec_times = dict()
        self.read_inputs(log_time=exec_times)
        path = os.path.join('outputs' , self.settings['folder'], 'generated_logs')
        # read generated logs
        generated_logs = list()
        for _, _, files in os.walk(path):
            for f in files:
                log = pd.read_csv(os.path.join(path, f))
                log = log[['run_num', 'caseid', 'task', 'role',
                           'start_timestamp', 'end_timestamp', ]]
                if is_numeric_dtype(log['caseid']):
                    log['caseid'] = 'Case'+ log['caseid'].astype(str)
                log['source'] = f.replace(".csv", ".h5")
                generated_logs.append(log)
        if generated_logs:
            generated_logs = pd.concat(generated_logs, axis=0, ignore_index=True)
            generated_logs.rename(columns={'role': 'user'}, inplace=True)
            generated_logs['start_timestamp'] =  pd.to_datetime(
                generated_logs['start_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f') 
            generated_logs['end_timestamp'] =  pd.to_datetime(
                generated_logs['end_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f') 
        # read simulated logs
        path = os.path.join('outputs' , self.settings['folder'], 'sim_logs')
        if os.path.exists(path):
            simulated_logs = list()
            for _, _, files in os.walk(path):
                for f in files:
                    splited = f.split('_')
                    log_name = '_'.join(splited[:-1])
                    rep = int(splited[-1].split('.')[0]) - 1
                    rep_results = pd.read_csv(os.path.join(path, f),
                                              dtype={'caseid': object})
                    rep_results['caseid'] = 'Case' + rep_results['caseid']
                    rep_results['run_num'] = rep
                    rep_results['source'] = log_name+'_training.bpmn'
                    simulated_logs.append(rep_results)
            simulated_logs = pd.concat(simulated_logs, axis=0, ignore_index=True)
            simulated_logs.rename(columns={'resource': 'user'}, inplace=True)
            simulated_logs['start_timestamp'] =  pd.to_datetime(
                simulated_logs['start_timestamp'], format='%Y-%m-%d %H:%M:%S.%f') 
            simulated_logs['end_timestamp'] =  pd.to_datetime(
                simulated_logs['end_timestamp'], format='%Y-%m-%d %H:%M:%S.%f') 
            if generated_logs:
                results = pd.concat([generated_logs, simulated_logs], axis=0,
                                    ignore_index=True)
            else:
                results = simulated_logs
        else:
            results = generated_logs
        results = results.append(self.log_test, ignore_index=True)
        self.evaluate_predict_log(results, self.settings, log_time=exec_times)
        self.save_times(exec_times, self.settings)
        
    def create_output_folders(self) -> None:
        # Output folder creation
        if not os.path.exists(self.settings['output']):
            os.makedirs(self.settings['output'])

    @timeit
    def read_inputs(self, **kwargs) -> None:
        # Event log reading
        self.log_test = lr.LogReader(
            os.path.join(self.settings['input'], self.settings['log_name'],
                         self.settings['log_name'] + '_testing.csv'),
            self.settings['read_options'])
        self.log_test = pd.DataFrame(self.log_test.data)
        if is_numeric_dtype(self.log_test['caseid']):
            self.log_test['caseid'] = 'Case'+ self.log_test['caseid'].astype(str)
        self.log_test['run_num'] = 0
        self.log_test['source'] = 'log'
        self.settings['num_cases'] = len(self.log_test.caseid.unique())
        # self.settings['num_cases'] = 2
    
    @timeit
    def find_models(self, **kwargs) -> list():
        path = os.path.join(self.settings['input'], self.settings['log_name'])
        files_filtered = list()
        for path, _, files in os.walk(path):
            for f in files:
                _, file_extension = os.path.splitext(f)
                if file_extension in ['.h5', '.bpmn']:
                    files_filtered.append(os.path.join(path, f))
        return files_filtered


    def generate_log(self, extension, model):
        generator = self._get_generator(extension)
        return generator(model)
    
    def _get_generator(self, extension):
        if extension == '.h5':
            return self._generate_log
        elif extension == '.bpmn':
            return self._simulate_model
        else:
            raise ValueError(extension)          

    def _simulate_model(self, model) -> None:
        print('-- Executing BIMP Simulations --')
        modified_model = self.modify_simulation_model(model)
        if not os.path.exists(os.path.join(self.settings['output'], 'sim_logs')):
            os.makedirs(os.path.join(self.settings['output'], 'sim_logs'))
        predictions = list()
        for rep in range(self.settings['repetitions']):
            print("Experiment #" + str(rep + 1))
            try:
                self.execute_simulator(self.settings, rep, modified_model)
                rep_results = pd.read_csv(
                    os.path.join(self.settings['output'], 'sim_logs',
                                 self.settings['log_name']+'_'+str(rep+1)+'.csv'),
                    dtype={'caseid': object})
                rep_results['caseid'] = 'Case' + rep_results['caseid']
                rep_results['run_num'] = rep
                rep_results['source'] = os.path.split(model)[1]
                predictions.append(rep_results)
            except Exception as e:
                print(e)
                break
        predictions = pd.concat(predictions, axis=0, ignore_index=True)
        predictions.rename(columns={'resource': 'user'}, inplace=True)
        predictions['start_timestamp'] =  pd.to_datetime(
            predictions['start_timestamp'], format='%Y-%m-%d %H:%M:%S.%f') 
        predictions['end_timestamp'] =  pd.to_datetime(
            predictions['end_timestamp'], format='%Y-%m-%d %H:%M:%S.%f') 
        return predictions

    def _generate_log(self, model) -> None:
        print('-- Generating Traces --')
        output_route = os.path.join(self.settings['output'], 'generated_logs')
        if not os.path.exists(output_route):
            os.makedirs(output_route)
        folder, model_file = os.path.split(model)
        generator = mp.ModelPredictor({
            'model_file': model_file,
            'folder': folder,
            'output_route': output_route,
            'num_cases': self.settings['num_cases'],
            'rep': self.settings['repetitions']})
        predictions = generator.predictions.copy()
        predictions = predictions[['run_num', 'caseid', 'task', 'role',
                                    'start_timestamp', 'end_timestamp', ]]
        predictions['source'] = model_file
        predictions.rename(columns={'role': 'user'}, inplace=True)
        return predictions

    @timeit
    def evaluate_predict_log(self, data, parms, **kwargs) -> float:
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator(parms['read_options']['one_timestamp'])
        exp_desc = pd.DataFrame([exp_desc])
        # dl = evaluator.measure('dl', data)
        # print(dl.dl.mean())
        # exp_desc = pd.concat([exp_desc]*len(dl), ignore_index=True)
        # dl = pd.concat([dl, exp_desc], axis=1).to_dict('records')
        # self.save_results(dl, 'dl', parms)
        # els = evaluator.measure('els', data)
        # mean_els = els.els.mean()
        # els = pd.concat([els, exp_desc], axis=1).to_dict('records')
        # self.save_results(els, 'els', parms)
        mae = evaluator.measure('mae_log', data)
        print(mae.mae_log.mean())
        exp_desc = pd.concat([exp_desc]*len(mae), ignore_index=True)
        mae = pd.concat([mae, exp_desc], axis=1).to_dict('records')
        self.save_results(mae, 'mae', parms)
        return mae

# =============================================================================
# Support
# =============================================================================

    def modify_simulation_model(self, model):
        """Modifies the number of instances of the BIMP simulation model
        to be equal to the number of instances in the testing log"""
        mydoc = minidom.parse(model)
        items = mydoc.getElementsByTagName('qbp:processSimulationInfo')
        items[0].attributes['processInstances'].value = str(
            self.settings['num_cases'])
        new_model_path = os.path.join(self.settings['output'],
                                      os.path.split(model)[1])
        with open(new_model_path, 'wb') as f:
            f.write(mydoc.toxml().encode('utf-8'))
        f.close()
        return new_model_path

    @staticmethod
    def save_times(times, settings):
        times = [{**{'output': settings['output']}, **times}]
        log_file = os.path.join('outputs', 'execution_times.csv')
        if not os.path.exists(log_file):
                open(log_file, 'w').close()
        if os.path.getsize(log_file) > 0:
            sup.create_csv_file(times, log_file, mode='a')
        else:
            sup.create_csv_file_header(times, log_file)

    @staticmethod
    def clean_parameters(parms):
        exp_desc = parms.copy()
        exp_desc.pop('read_options', None)
        exp_desc.pop('input', None)
        exp_desc.pop('bimp_path', None)
        exp_desc.pop('folder', None)
        exp_desc.pop('action', None)
        exp_desc.pop('repetitions', None)
        return exp_desc

    @staticmethod
    def save_results(measurements, feature, parms):
        if measurements:
            if os.path.exists(os.path.join(
                    'outputs', feature+'_'+'.csv')):
                sup.create_csv_file(
                    measurements,
                    os.path.join('outputs',
                                 feature+'_'+'.csv'),
                    mode='a')
            else:
                sup.create_csv_file_header(
                    measurements,
                    os.path.join('outputs',
                                 feature+'_'+'.csv'))

# =============================================================================
# External tools calling
# =============================================================================

    @staticmethod
    def execute_simulator(settings, rep, model):
        """Executes BIMP Simulations.
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        args = ['java', '-jar', settings['bimp_path'],
                os.path.join(model),
                '-csv',
                os.path.join(settings['output'], 'sim_logs',
                             settings['log_name']+'_'+str(rep+1)+'.csv')]
        subprocess.call(args)

        