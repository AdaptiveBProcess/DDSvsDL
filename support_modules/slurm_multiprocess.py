# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:49:25 2020
This module creates jobs and controls their
execution status automatically for slurm HPC
@author: Manuel Camargo
"""
import subprocess
import re
import os
import time
from support_modules import support as sup
import numpy as np
import shutil
from collections import defaultdict


from enum import Enum

# =============================================================================
# Job object
# =============================================================================
class St(Enum):
    HOLDING = 1
    SUBMITTED = 2
    PENDING = 3
    STARTED = 4
    COMPLETED = 5
    CANCELLED = 6

class Job():
    id_index = defaultdict(list)

    def __init__(self, args):
        """constructor"""
        self.__id = sup.folder_id()
        self.__status = St.HOLDING
        self.__worker_id = None
        self.__args = args
        Job.id_index[self.__id].append(self)

    def set_worker(self, worker_id):
        self.__worker_id = worker_id

    def set_status(self, status):
        self.__status = status

    def set_args(self, args):
        self.__args = args

    def get_id(self):
        return self.__id

    def get_status(self):
        return self.__status

    def get_args(self):
        return self.__args

    def get_worker(self):
        return self.__worker_id

    @classmethod
    def find_by_id(cls, id):
        return Job.id_index[id][0]

# =============================================================================
#  Jobs controller
# =============================================================================
class HPC_Multiprocess():
    def __init__(self, conn, args, control_output, output_folder, workers_num, timeout=1):
        """constructor"""
        self.jobs_folder = os.path.join(control_output, 'mp_jobs_files')
        self.stdout_folder = os.path.join(control_output, 'stdout')
        self.output_folder = output_folder
        self.conn = conn
        self.args = args
        self.workers_num = workers_num
        self.timeout = timeout

    def parallelize(self):
        self.clean_folder(self.jobs_folder)
        self.clean_folder(self.stdout_folder)
        # create queue
        self.queue = [Job(arg) for arg in self.args]
        self.mannage_queue()
        shutil.rmtree(self.jobs_folder)
        shutil.rmtree(self.stdout_folder)

# =============================================================================
# Create worker
# =============================================================================

    def create_worker(self, job_id):
        job = Job.find_by_id(job_id)
        exp_name = 'worker'
        default = ['#!/bin/bash',
                   '#SBATCH --partition='+self.conn['partition'],
                   '#SBATCH -J '+ exp_name,
                   '#SBATCH --output='+('"'+os.path.join(self.stdout_folder,'slurm-%j.out'+'"')),
                   '#SBATCH -N 1',
                   '#SBATCH --mem='+self.conn['mem'],
                   '#SBATCH -t 72:00:00',
                   'module load cuda/10.0',
                   'module load python/3.6.3/virtenv',
                   'source activate ' + self.conn['env']
                   ]
        def format_option(short, parm):
            return (' -'+short+' None'
                    if parm in [None, 'nan', '', np.nan]
                    else ' -'+short+' '+str(parm))

        options = 'python '+self.conn['script']
        for k, v in job.get_args().items():
            options += format_option(k, v)
        options += format_option('o', self.output_folder)
        # options += ' -a training'
        default.append(options)
        file_name = os.path.join(self.jobs_folder, sup.folder_id())
        sup.create_text_file(default, file_name)
        return self.submit_job(file_name)

    @staticmethod
    def submit_job(file):
        print(" -- submit batches --")
        args = ['sbatch', file]
        output = subprocess.run(args, capture_output=True).stdout
        prog = re.compile('(\d+)')
        results = prog.findall(output.decode("utf-8"))
        return results[0]

    def update_status(self):
        active_jobs = [job for job in self.queue if job.get_status() != St.HOLDING]
        for job in active_jobs:
            stdout_path = os.path.join(self.stdout_folder,'slurm-'+ job.get_worker() +'.out')
            if job.get_status() == St.SUBMITTED:
                if os.path.exists(stdout_path):
                    job.set_status(St.PENDING)
            elif job.get_status() == St.PENDING:
                if os.path.getsize(stdout_path) > 0:
                    job.set_status(St.STARTED)
            elif job.get_status() == St.STARTED:
                with open(stdout_path, 'r') as f:
                    last_line = self.tail(f, lines=1, _buffer=4098)[0]
                if 'COMPLETED' in last_line:
                    job.set_status(St.COMPLETED)
                if ('CANCELLED' in last_line) or ('Error' in last_line):
                    job.set_status(St.CANCELLED)

    def mannage_queue(self):
        completed = list()
        occupied_workers = 0
        completed = list()
        while self.queue:
            # update jobs status
            self.update_status()
            # Check cancelled jobs
            cancelled = [job for job in self.queue if job.get_status() == St.CANCELLED]
            if cancelled:
                raise Exception('Subprocesses interrupted: '+', '.join(
                                                [job.get_id() for job in cancelled]))
            # Check completed jobs
            completed.extend([job.get_worker() for job in self.queue if job.get_status() == St.COMPLETED])
            # Update queue
            self.queue = [
                job for job in self.queue if job.get_status() not in [St.COMPLETED,
                                                                St.CANCELLED]]
            occupied_workers = len([
                job for job in self.queue if job.get_status() not in [St.COMPLETED,
                                                                St.HOLDING,
                                                                St.CANCELLED]])
            holding_jobs = [job.get_id() for job in self.queue if job.get_status() == St.HOLDING]
            if occupied_workers <= self.workers_num and len(holding_jobs) > 0:
                # Create as much workers as nedded or possible
                num_jobs = min((self.workers_num - occupied_workers), len(holding_jobs))
                for job_id in holding_jobs[:num_jobs]:
                    worker_id = self.create_worker(job_id)
                    job = Job.find_by_id(job_id)
                    job.set_worker(worker_id)
                    job.set_status(St.SUBMITTED)
            # [print('QUEUE', job.get_id(), job.get_status(), job.get_worker(), sep=' ') for job in self.queue]
            print('Queue:', len(self.queue),'Completed:',len(completed), sep=' ')
            time.sleep(self.timeout)
        # [print('COMP', job, sep=' ') for job in completed]
# =============================================================================
# Support methods
# =============================================================================
    @staticmethod
    def create_file_list(path):
        file_list = list()
        for root, dirs, files in os.walk(path):
            for f in files:
                file_list.append(f)
        return file_list

    @staticmethod
    def clean_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        # clean folder
        for _, _, files in os.walk(folder):
            for file in files:
                os.unlink(os.path.join(folder, file))

    @staticmethod
    def tail(f, lines=1, _buffer=4098):
        """Tail a file and get X lines from the end"""
        # place holder for the lines found
        lines_found = []
        # block counter will be multiplied by buffer
        # to get the block size from the end
        block_counter = -1
        # loop until we find X lines
        while len(lines_found) < lines:
            try:
                f.seek(block_counter * _buffer, os.SEEK_END)
            except IOError:  # either file is too small, or too many lines requested
                f.seek(0)
                lines_found = f.readlines()
                break

            lines_found = f.readlines()

            # decrement the block counter to get the next X bytes
            block_counter -= 1
        return lines_found[-lines:]