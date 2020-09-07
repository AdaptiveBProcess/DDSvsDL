# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:25:10 2019

@author: Manuel Camargo
"""
import os
import sys
import getopt
import evaluator as ev

from support_modules import support as sup

# =============================================================================
# Main function
# =============================================================================

def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-l': 'log_name', '-r': 'repetitions', '-a': 'action', '-f': 'folder'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)


# --setup--
def main(argv):
    """Main aplication method"""
    parameters = dict()
    """ Sets the app general settings"""
    column_names = {'Case ID': 'caseid', 'Activity': 'task',
                    'lifecycle:transition': 'event_type', 'Resource': 'user'}
    # Event-log reading options
    parameters['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                'column_names': column_names,
                                'one_timestamp': False,
                                'filter_d_attrib': True,
                                'ns_include': True}
    # Folders structure
    parameters['input'] = 'input_files'
    parameters['output'] = os.path.join('outputs', sup.folder_id())
    # External tools routes
    parameters['bimp_path'] = os.path.join(
        'generators', 'external_tools', 'bimp', 'qbp-simulator-engine.jar')
    parameters['action'] = 'execute'
    parameters['folder'] = '20200804_142204829211'
    
    # Parameters setting manual fixed or catched by console
    if not argv:
        # Event-log parameters
        parameters['log_name'] = 'Production'
        # Specific model training parameters
        parameters['repetitions'] = 1  # keras lstm implementation 1 cpu,2 gpu
    else:
        # Catch parameters by console
        try:
            opts, _ = getopt.getopt(
                argv, "hl:r:a:f:", ['log_name=', 'repetitions=', 'action=', 'folder='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['repetitions']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
            
    print(parameters)
    evaluator = ev.Evaluator(parameters)
    if parameters['action'] == 'evaluate':
        evaluator.read_and_evaluate()
    else:
        evaluator.execute_pipeline()


if __name__ == "__main__":
    main(sys.argv[1:])
