import os
import argparse

from iquaflow.datasets import DSWrapper, DSModifier
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution


def main(args):
    #Define name of IQF experiment
    experiment_name = args.experiment_name

    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    data_augmentation_flag = "True"

    earlystopper_patience = args.earlystop
    lr_reducer_patience = args.lrreduce

    for product in ['L3', 'RR']:

        tag = f"{product}_{args.tag}"
        #DS wrapper is the class that encapsulate a dataset
        root_path = args.dataset_path
        trainds = DSWrapper(data_path=root_path+product+"/train/")
        valds = DSWrapper(data_path=root_path+product+"/val/")
        testds = DSWrapper(data_path=root_path+product+"/test/")

        # #Define path of the training script
        python_ml_script_path = "main.py"

        base_modifier = DSModifier()
        base_modifier._toggle_on_symlink_for_base_modifier()

        ds_modifiers_list = [ base_modifier ]

        
        task = PythonScriptTaskExecution( model_script_path = python_ml_script_path, tmp_dir = "tmp")

        #Experiment definition, pass as arguments all the components defined beforehand
        experiment = ExperimentSetup(
            experiment_name=experiment_name,
            task_instance=task,
            ref_dsw_train=trainds,
            ref_dsw_val=valds,
            ref_dsw_test=testds,
            ds_modifiers_list=ds_modifiers_list,
            repetitions=args.repetitions,
            mlflow_monitoring  = True,
            extra_train_params={'product': [product], 
                                'tag': [tag], 
                                'epochs': [epochs], 
                                'learning_rate': [learning_rate], 
                                'batch_size': [batch_size], 
                                'data_augmentation_flag': [data_augmentation_flag],
                                'earlystopper_patience': [earlystopper_patience],
                                'lr_reducer_patience': [lr_reducer_patience]})

        experiment.execute()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Run the building segmentation experiment')
    parser.add_argument('--experiment-name',
                        dest='experiment_name',
                        help='The name of the experiment',
                        required=True,
                        type=str)
    parser.add_argument('--dataset-path',
                        dest='dataset_path',
                        help='Path to the main folder of the dataset',
                        required=True,
                        type=str)
    parser.add_argument('--epochs',
                        dest='epochs',
                        help='Maximum number of training epochs',
                        required=False,
                        default=100,
                        type=int)
    parser.add_argument('--learning-rate',
                        dest='lr',
                        help='Initial learning rate',
                        required=False,
                        default=0.001,
                        type=float)
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='Batch size',
                        required=False,
                        default=16,
                        type=int)
    parser.add_argument('--earlystopper-patience',
                        dest='earlystop',
                        help='Patience of early stopper',
                        required=False,
                        default=25,
                        type=int)
    parser.add_argument('--lrreducer-patience',
                        dest='lrreduce',
                        help='Patience of the learning rate reducer',
                        required=False,
                        default=9,
                        type=int)
    parser.add_argument('--repetitions',
                        dest='repetitions',
                        help='The number of time to repeat the experiment',
                        required=False,
                        default=5,
                        type=int)
    parser.add_argument('--tag',
                        dest='tag',
                        help='A tag used for experiment tracking',
                        required=False,
                        default='v1',
                        type=str)
        
    
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    main(args)
