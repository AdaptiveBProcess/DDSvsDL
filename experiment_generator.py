# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:08:16 2019

@author: Manuel Camargo
"""
import itertools
from support_modules import support as sup
import os
import random
import time
import pandas as pd
import numpy as np

# =============================================================================
#  Support
# =============================================================================


def create_file_list(path):
    file_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            file_list.append(f)
    return file_list

# =============================================================================
# Sbatch files creator
# =============================================================================


def sbatch_creator(configs):
    for i, _ in enumerate(configs):
        exp_name = configs[i]['log'].lower()[:5]
        if imp == 2:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=gpu',
                       '#SBATCH --gres=gpu:tesla:1',
                       '#SBATCH -J ' + exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --cpus-per-task=20',
                       '#SBATCH --mem=14000',
                       '#SBATCH -t 72:00:00',
                       'module load cuda/10.0',
                       'module load java-1.8.0_40',
                       'module load python/3.6.3/virtenv',
                       'source activate lstm_exp_v3_gpu'
                       ]
        else:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=amd',
                       '#SBATCH -J ' + exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --cpus-per-task=20',
                       '#SBATCH --mem=14000',
                       '#SBATCH -t 72:00:00',
                       'module load cuda/10.0',
                       'module load java-1.8.0_40',
                       'module load python/3.6.3/virtenv',
                       'source activate lstm_exp_v3'
                       ]

        def format_option(short, parm):
            return (' -'+short+' None'
                    if configs[i][parm] in [None, 'nan', '', np.nan]
                    else ' -'+short+' '+str(configs[i][parm]))

        options = 'python evaluator_parameters.py'
        options += ' -a '+action
        options += format_option('l', 'log')
        if action == 'evaluate':
            options += format_option('f', 'folder')
        else:
            options += format_option('r', 'rep')
        
        default.append(options)
        file_name = sup.folder_id() + str(i)
        sup.create_text_file(default, os.path.join(output_folder, file_name))

# =============================================================================
# Sbatch files submission
# =============================================================================


def sbatch_submit(in_batch, bsize=20):
    file_list = create_file_list(output_folder)
    print('Number of experiments:', len(file_list), sep=' ')
    for i, _ in enumerate(file_list):
        if in_batch:
            if (i % bsize) == 0:
                time.sleep(20)
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
            else:
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
        else:
            os.system('sbatch '+os.path.join(output_folder, file_list[i]))

# =============================================================================
# Kernel
# =============================================================================


# create output folder
output_folder = 'jobs_files'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# clean folder
for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

# parameters definition

# s2, sh
imp = 1  # keras lstm implementation 1 cpu, 2 gpu
configs = list()

action = 'execute'

if action == 'evaluate':
    configs = [{'log': 'PurchasingExample', 'folder': '20200804_141914344697'},
            {'log': 'ConsultaDataMining201618', 'folder': '20200804_141914344704'},
            {'log': 'BPI_Challenge_2012_W_Two_TS', 'folder': '20200804_145640792071'}]
    
else:
    logs = ['inter_Production', 'inter_PurchasingExample', 'inter_ConsultaDataMining201618',
            'inter_BPI_Challenge_2012_W_Two_TS']
    for log in logs:
        # configs definition
        configs.append({'rep': 10, 'log': log})
        # sbatch creation
sbatch_creator(configs)
# submission
sbatch_submit(True)
