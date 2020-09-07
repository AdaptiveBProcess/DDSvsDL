# Reproductibility package for the DDS vs DL article

This repository contains the source code for reproducing the results of the paper "Discovering Generative Models from Event Logs: Data-driven Simulation vs. Deep Learning."  This paper compared two families of generative process simulation models developed in previous work: data-driven simulation models and deep learning models. Until now, these two approaches have evolved independently, and their relative performance has not been studied.

## Prerequisites

To execute this code you just need to install Anaconda in your system, and create an environment using the *environment.yml* specification provided in the repository.

## Datasets
 
The evaluator assumes the input is composed by a case identifier, an activity label, a resource attribute (indicating which resource performed the activity), and two timestamps: the start timestamp and the end timestamp. The resource attribute is required in order to discover the available resource pools, their timetables, and the mapping between activities and resource pools, which are a required element in a BPS model. We require both start and endtimestamps for each activity instance, in order to compute the processing time of activities, which is also a required element in a simulation model.

### Evaluator inputs

In the input_files folder, it is required a subfolder named with the event-log name to evaluate. In this subfolder, must be located the testing partition used to evaluate the accuracy of the logs, a subfolder called generative_models, and another called simulation_models. The generative_models folder contains the best LSTM and GRU models found with the [DeepGenerator](https://github.com/AdaptiveBProcess/GenerativeLSTM) tool, and the simulation_models folder contains the best data-driven simulation models found with the [Simod](https://github.com/AdaptiveBProcess/Simod) tool. For ease of reproduction, we provide in this repository the folders and models used in the evaluation of the article. Likewise, we provide the complete event logs in the file "complete_logs.zip."

```
input_files
+-- Production
|   +-- Production_testing.csv
|   +-- generative_models
|   |   +-- model_folder
|   |   |   +-- parameters
|   |   |   +-- model.h5
|   +-- simulation_models
|   |   +-- model_folder
|   |   |   +-- Production.bpmn
```

## Running the script

Once created the environment, you can evaluate the models, specifying the following parameters in the evaluator_parameters.py module, or by command line specifying the required activity (-a) as 'execute' followed by the name of the event log (-l), and the number of repetitions (-r):

```
(env) C:\sc_evaluator>python evaluator_parameters.py -a execute -l Production -r 10
```

## Simod:
See [Simod](https://github.com/AdaptiveBProcess/Simod).

## DeepGenerator:
See [DeepGenerator](https://github.com/AdaptiveBProcess/GenerativeLSTM).

## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
