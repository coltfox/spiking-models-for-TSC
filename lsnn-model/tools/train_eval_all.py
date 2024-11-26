import argparse

import os
import numpy as np
from consts.dir_consts import DRC, BASE_DIR
from consts.run_time_consts import RTC
from consts.exp_consts import EXC
from src.pyt_train_eval_lsnn_nspk import PTNspkTrainEvalModel
from src.pyt_train_eval_lsnn_and_lsnn_nhdn_model import PTTrainEvalModel
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils

OTP_DIR = f"{BASE_DIR}/results/"
DATASETS = ["ECG5000", "FORDA", "FORDB", "WAFER", "EQUAKES"]
MODELS = ["LSNN", "BW_NSPK", "LSNN_NHDN"]

def train_eval_all_models(dataset, epochs, use_all_combs=False):
   for model in MODELS:
      train_eval_all(model, dataset, epochs, use_all_combs)

def train_eval_all(model, dataset, epochs, use_all_combs=False):
    # Don't need to save any files
    otp_dir = f"{OTP_DIR}/{dataset}-{model}/"
    os.makedirs(otp_dir, exist_ok=True)
    setup_logging(otp_dir, model, dataset)

    max_acc, avg_acc = 0.0, 0.0

    configure_constants(model, dataset)

    if use_all_combs:
        max_acc, avg_acc = call_all_combs(otp_dir, dataset, epochs)
        log.INFO("All combination experiments done!")
    else:
       acc = call_one_combination(otp_dir, dataset, epochs)
       max_acc, avg_acc = acc, acc
    
    log.INFO(f"Max acc: {max_acc}")
    log.INFO(f"Avg acc: {avg_acc}")

def call_all_combs(otp_dir, dataset, epochs):
    exu = ExpUtils()
    
    for seed in [6, 9, 100]:
        log.INFO(f"Running experiments on seed {seed}")
        RTC.SEED = seed
        
        if ("NSPK" in RTC.PYTORCH_MODEL_NAME):
            return call_all_combs_nspk(exu, otp_dir, dataset, epochs)
        
        return call_all_combs_lsnn(exu, otp_dir, dataset, epochs)

def call_all_combs_lsnn(exu, otp_dir, dataset, epochs):
    accuracies = []
    all_combs = list(exu.get_combinations_for_lsnn_model(EXC))
    num_combs = len(all_combs)

    for i, comb in enumerate(all_combs):
      log.INFO(f"Running hyper-parameter combination {i + 1}/{num_combs}")
      RTC.PYTORCH_LR = comb[0]
      RTC.PYTORCH_TAU_CUR = comb[1]
      RTC.PYTORCH_TAU_VOL = comb[2]
      RTC.PYTORCH_VOL_THR = comb[3]
      RTC.PYTORCH_NEURON_GAIN = comb[4]
      RTC.PYTORCH_NEURON_BIAS = comb[5]

      acc = call_one_combination(otp_dir, dataset, epochs)
      accuracies.append(acc)
    
    max_acc = np.max(accuracies)
    avg_acc = np.mean(accuracies)

    return max_acc, avg_acc

def call_all_combs_nspk(exu, otp_dir, dataset, epochs):
    accuracies = []
    all_combs = list(exu.get_combinations_for_lsnn_nspk(EXC))
    num_combs = len(all_combs)

    for i, comb in enumerate(all_combs):
      log.INFO(f"Running hyper-parameter combination {i + 1}/{num_combs}")
      RTC.ORDER = comb[0]
      RTC.THETA = comb[1]
      RTC.PYTORCH_LR = comb[2]

      acc = call_one_combination(otp_dir, dataset, epochs)
      accuracies.append(acc)
    
    max_acc = np.max(accuracies)
    avg_acc = np.mean(accuracies)

    return max_acc, avg_acc

def call_one_combination(otp_dir, dataset, epochs):
    if "NSPK" in otp_dir:
        trainer = PTNspkTrainEvalModel(dataset, RTC)
    else:
        trainer = PTTrainEvalModel(dataset, RTC)

    log.INFO("Starting the PyTorch training...")
    trainer.train_model(epochs, ldn_path=otp_dir)

    log.INFO("Training done, now finally evaluating on the entire test set...")
    acc, _ = trainer.evaluate_model(
      num_samples=EXC.NUM_TEST_SAMPLES[dataset], ldn_path=otp_dir)
    
    log.INFO("Evaluation done. Test accuracy: {}".format(acc))

    cleanup_sigs(otp_dir)
    log.INFO("Files removed... Experiment done!")
    # log.RESET()

    return acc

def cleanup_sigs(otp_dir):
    os.remove(otp_dir+"/test_X_ldn_sigs.p")
    os.remove(otp_dir+"/test_Y.p")
    os.remove(otp_dir+"/train_X_ldn_sigs.p")
    os.remove(otp_dir+"/train_Y.p")

def setup_logging(otp_dir, model, dataset):
  log.configure_log_handler(f"{otp_dir}/{model}-{dataset}-train-eval-{ExpUtils().get_timestamp()}")
  keys = list(vars(RTC).keys())
  log.INFO("#"*30 + " C O N F I G " + "#"*30)
  for key in keys:
    log.INFO("{0}: {1}".format(key, getattr(RTC, key)))
  log.INFO("#"*70)

def configure_constants(model, dataset):
    """Set the correct runtime and experiment constants for the dataset"""

    RTC.PYTORCH_MODEL_NAME = model
    
    if dataset in ["ECG5000", "WAFER"]:
        EXC.PYTORCH_NEURON_GAIN_LST = [1, 2, 4]
        EXC.PYTORCH_NEURON_BIAS_LST = [0, 0.5]
        EXC.PYTORCH_TAU_CUR_LST = [5e-3, 10e-3, 15e-3]
        EXC.PYTORCH_TAU_VOL_LST = [10e-3, 20e-3, 30e-3]
        EXC.PYTORCH_VOL_THR_LST = [1, 1.5]
        EXC.PYTORCH_LR_LST = [0.01, 0.05, 0.005]
    elif dataset in ["EQUAKES", "FORDA", "FORDB"]:
        EXC.PYTORCH_NEURON_GAIN_LST = [2, 4]
        EXC.PYTORCH_NEURON_BIAS_LST = [0, 0.5]
        EXC.PYTORCH_TAU_CUR_LST = [5e-3, 10e-3]
        EXC.PYTORCH_TAU_VOL_LST = [20e-3, 30e-3]
        EXC.PYTORCH_VOL_THR_LST = [1, 1.5]
        EXC.PYTORCH_LR_LST = [0.01, 0.005]

    if dataset == "ECG5000":
        RTC.TEST_EVAL_SIZE = 4500
        RTC.BATCH_SIZE = 50
    elif dataset == "FORDA":
        RTC.TEST_EVAL_SIZE = 1320
        RTC.BATCH_SIZE = 40
    elif dataset == "FORDB":
        RTC.TEST_EVAL_SIZE = 810
        RTC.BATCH_SIZE = 18
    elif dataset == "WAFER":
        RTC.TEST_EVAL_SIZE = 6150
        RTC.BATCH_SIZE = 50
    elif dataset == "EQUAKES":
        RTC.TEST_EVAL_SIZE = 138
        RTC.BATCH_SIZE = 23


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Which model?")
parser.add_argument("--dataset", type=str, required=True, help="Which dataset?")
parser.add_argument("--epochs", type=int, required=True, help="Training epochs?")
parser.add_argument("--is_all_combs", type=int, required=False, choices=[0, 1],
                    default=0, help="Search over all hyper-params combinations?")

args = parser.parse_args()

if args.model == "all":
    train_eval_all_models(args.dataset, args.epochs, args.is_all_combs)
else:
    train_eval_all(args.model, args.dataset, args.epochs, args.is_all_combs)