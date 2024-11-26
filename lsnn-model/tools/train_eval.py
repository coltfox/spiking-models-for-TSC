import argparse

import os
import numpy as np
from consts.dir_consts import DRC, BASE_DIR, RESULTS_DIR
from consts.run_time_consts import RTC
from consts.exp_consts import EXC
from src.pyt_train_eval_lsnn_nspk import PTNspkTrainEvalModel
from src.pyt_train_eval_lsnn_and_lsnn_nhdn_model import PTTrainEvalModel
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from multiprocessing import Process, Value, Array, cpu_count
import math
import logging

class TrainEval:
    """Trains and evaluates a model with a particular dataset across all combinations of hyper-parameters"""
    SEEDS = [6, 9, 100]
    
    def __init__(self, model, dataset, epochs, num_cores=4):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.otp_dir = f"{RESULTS_DIR}/{dataset}-{model}/"
        self.exu = ExpUtils()
        
        if num_cores > cpu_count():
            raise ValueError(f"Cannot train and evaluate across {num_cores} cores, only {cpu_count()} available!")
        
        self.num_procceses = num_cores
        
        # Need to use a different logger to avoid conflicts with subprocesses
        self.log = logging.getLogger(str(id(self)))
    
    def train_eval_all_combs(self):
        """Train and evaluate the model across all combinations of hyper-parameters"""
        self._configure_constants()
        os.makedirs(self.otp_dir, exist_ok=True)
        self._setup_logging()
        # self._setup_child_logging(self.otp_dir)
        all_combs = self._get_all_combinations()
        num_combs = len(all_combs)
        combs_per_proc = math.ceil(num_combs / self.num_procceses)
        split_combs = self._split_combinations(all_combs, combs_per_proc)
        
        accuracies = []
        
        for seed in self.SEEDS:
            self.log.info(f"Running experiments on seed {seed}")
            RTC.SEED = seed
            
            self.log.info(f"Running {num_combs} hyper-parameter combinations in groups of {combs_per_proc}...")
            num_groups = len(split_combs)
            
            if num_groups > cpu_count():
                raise ValueError(f"Failed to create {num_groups} processes of {combs_per_proc} combinations. Only {cpu_count()} cores available!")
            
            # Values become available after all processes finish
            accuracies_seed = Array('d', num_combs)
            processes = []
            
            # for i, comb in enumerate(all_combs):
            #     self.log.info(f"Starting process for hyper-parameter combination {i + 1}/{num_combs}")
            #     self._set_combination(comb)
            #     acc = self.train_eval_single()
            #     accuracies.append(acc)
            
            for i, comb_group in enumerate(split_combs):
                self.log.info(f"Starting process for hyper-parameter combination group {i + 1}/{num_groups}")
                
                proc = Process(target=TrainEval._train_eval_group, args=(self, comb_group, accuracies_seed, i))
                proc.start()
                
                processes.append(proc)
            
            for i, proc in enumerate(processes):
                self.log.info(f"Waiting for procces {i + 1}/{len(processes)} ({proc.pid})")
                proc.join()
            
            accuracies.extend(list(accuracies_seed))
            
            self.log.info(f"All processes finished for seed {seed}.")
        
        self.log.info("All combinations for all seeds have run successfully. Experiment complete!")
        
        max_acc = np.max(accuracies)
        avg_acc = np.mean(accuracies)
        
        self.log.info("#"*30 + " R E S U L T S " + "#"*30)
        
        self.log.info(f"Maximum accuracy: {max_acc}")
        self.log.info(f"Avgerage accuracy: {avg_acc}")
    
    def _configure_constants(self):
        """Set the correct runtime and experiment constants for the dataset"""

        RTC.PYTORCH_MODEL_NAME = self.model
        
        if self.dataset in ["ECG5000", "WAFER"]:
            EXC.PYTORCH_NEURON_GAIN_LST = [1, 2, 4]
            EXC.PYTORCH_NEURON_BIAS_LST = [0, 0.5]
            EXC.PYTORCH_TAU_CUR_LST = [5e-3, 10e-3, 15e-3]
            EXC.PYTORCH_TAU_VOL_LST = [10e-3, 20e-3, 30e-3]
            EXC.PYTORCH_VOL_THR_LST = [1, 1.5]
            EXC.PYTORCH_LR_LST = [0.01, 0.05, 0.005]
        elif self.dataset in ["EQUAKES", "FORDA", "FORDB"]:
            EXC.PYTORCH_NEURON_GAIN_LST = [2, 4]
            EXC.PYTORCH_NEURON_BIAS_LST = [0, 0.5]
            EXC.PYTORCH_TAU_CUR_LST = [5e-3, 10e-3]
            EXC.PYTORCH_TAU_VOL_LST = [20e-3, 30e-3]
            EXC.PYTORCH_VOL_THR_LST = [1, 1.5]
            EXC.PYTORCH_LR_LST = [0.01, 0.005]

        if self.dataset == "ECG5000":
            RTC.TEST_EVAL_SIZE = 4500
            RTC.BATCH_SIZE = 50
        elif self.dataset == "FORDA":
            RTC.TEST_EVAL_SIZE = 1320
            RTC.BATCH_SIZE = 40
        elif self.dataset == "FORDB":
            RTC.TEST_EVAL_SIZE = 810
            RTC.BATCH_SIZE = 18
        elif self.dataset == "WAFER":
            RTC.TEST_EVAL_SIZE = 6150
            RTC.BATCH_SIZE = 50
        elif self.dataset == "EQUAKES":
            RTC.TEST_EVAL_SIZE = 138
            RTC.BATCH_SIZE = 23
    
    def _get_all_combinations(self):
        """Get all combinations split by COMBS_PER_PROC"""
        if self.model == "BW_NSPK":
            return list(self.exu.get_combinations_for_lsnn_nspk(EXC))

        return list(self.exu.get_combinations_for_lsnn_model(EXC))

    def _split_combinations(self, combs, combs_per_proc):
        """Split combinations into COMBS_PER_PROC sections"""
        return [combs[i:i + combs_per_proc] for i in range(0, len(combs), combs_per_proc)]
    
    def _set_combination(self, comb):
        if self.model == "BW_NSPK":
            RTC.ORDER = comb[0]
            RTC.THETA = comb[1]
            RTC.PYTORCH_LR = comb[2]
        else:
            RTC.PYTORCH_LR = comb[0]
            RTC.PYTORCH_TAU_CUR = comb[1]
            RTC.PYTORCH_TAU_VOL = comb[2]
            RTC.PYTORCH_VOL_THR = comb[3]
            RTC.PYTORCH_NEURON_GAIN = comb[4]
            RTC.PYTORCH_NEURON_BIAS = comb[5]

    def _train_eval_group(self, comb_group, accuracies_arr, group_ind):
        """Train and evaluate group of combinations on a separate process. Store accuracy in accuracies array. Returns the process"""
        combs_dir = f"{self.otp_dir}/comb-group-{group_ind + 1}"
        os.makedirs(combs_dir, exist_ok=True)
        self._setup_child_logging(combs_dir)
        
        for i, comb in enumerate(comb_group):
            log.INFO(f"\nRUNNING COMBINATION {group_ind + i + 1}/{len(accuracies_arr)}\n")
            self._set_combination(comb)
            acc = self.train_eval_single(combs_dir)
            accuracies_arr[group_ind + i] = acc
        
        log.RESET()
    
    def train_eval_single(self, sigs_dir=None):
        """Train and evaluate using the current configuration of hyper-parameters"""
        sigs_dir = sigs_dir or self.otp_dir
        
        if self.model == "BW_NSPK":
            trainer = PTNspkTrainEvalModel(self.dataset, RTC)
        else:
            trainer = PTTrainEvalModel(self.dataset, RTC)

        log.INFO("Starting the PyTorch training...")
        trainer.train_model(self.epochs, ldn_path=sigs_dir)

        log.INFO("Training done, now finally evaluating on the entire test set...")
        acc, _ = trainer.evaluate_model(
        num_samples=RTC.TEST_EVAL_SIZE, ldn_path=sigs_dir)
        
        log.INFO("Evaluation done. Test accuracy: {}".format(acc))

        self.cleanup_sigs(sigs_dir)
        log.INFO("Files removed... Experiment done!")

        return acc
        
    def _setup_logging(self):
        """Setup logger for parent process"""
        self.log.setLevel(logging.DEBUG)
        
        log_file = f"{self.otp_dir}/results-{ExpUtils().get_timestamp()}.log"
        
        file_handler = logging.FileHandler(log_file)
        self.log.addHandler(file_handler)

        # If applicable, delete the existing log file to generate a fresh log file
        # during each execution
        if os.path.isfile(log_file):
            os.remove(log_file)

        # Create handler for logging the messages to a log file.
        log_handler = logging.FileHandler(log_file)
        log_handler.setLevel(logging.DEBUG)

        # Set the format of the log.
        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Add the Formatter to the Handler
        log_handler.setFormatter(log_formatter)

        # Add stream handler.
        self.log.addHandler(logging.StreamHandler())

        # Add the Handler to the Logger
        self.log.addHandler(log_handler)
    
    def _setup_child_logging(self, log_dir):
        """Setup logging for child process"""
        log.configure_log_handler(f"{log_dir}/{self.model}-{self.dataset}-train-eval-{ExpUtils().get_timestamp()}")
        keys = list(vars(RTC).keys())
        log.INFO("#"*30 + " C O N F I G " + "#"*30)
        for key in keys:
            log.INFO("{0}: {1}".format(key, getattr(RTC, key)))
        log.INFO("#"*70)
    
    # def _log_config(self):
    #     keys = list(vars(RTC).keys())
    #     self.log.info("#"*30 + " C O N F I G " + "#"*30)
    #     for key in keys:
    #         self.log.info("{0}: {1}".format(key, getattr(RTC, key)))
    #     self.log.info("#"*70)
    
    def cleanup_sigs(self, dir):
        os.remove(dir+"/test_X_ldn_sigs.p")
        os.remove(dir+"/test_Y.p")
        os.remove(dir+"/train_X_ldn_sigs.p")
        os.remove(dir+"/train_Y.p")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Which model?")
    parser.add_argument("--dataset", type=str, required=True, help="Which dataset?")
    parser.add_argument("--epochs", type=int, required=True, help="Training epochs?")
    parser.add_argument("--num_cores", type=int, required=True, help="Number of cores to split up the work across")
    parser.add_argument("--is_all_combs", type=int, required=False, choices=[0, 1],
                        default=0, help="Search over all hyper-params combinations?")

    args = parser.parse_args()
    
    if args.model == "all":
        for model in ["BW_NSPK", "LSNN", "LSNN_NHDN"]:
            train_eval = TrainEval(
                model=model,
                dataset=args.dataset,
                epochs=args.epochs,
                num_cores=args.num_cores
            )
            
            train_eval.train_eval_all_combs()
    else:
        train_eval = TrainEval(
            model=args.model,
            dataset=args.dataset,
            epochs=args.epochs,
            num_cores=args.num_cores
        )
        
        train_eval.train_eval_all_combs()