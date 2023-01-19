# spiking-models-for-TSC
Code to accompany our paper: Spiking Models for Univariate Time Series Classification

Two models are presented in the paper:
* Spiking Legendre Reservoir Computing model
* Legendre Spiking Neural Network model

Repository description:
* `slrc-model` directory contains all the code for SLRC model.
* `lsnn-model` directory contains all the code for LSNN model.
* `energy-consumption-analysis` directory contains all the code for measuring energy
	consumption on Loihi-1 and CPU.

To run the `slrc-model` and `lsnn-model` directories codes, you would be requried to
download the datasets (mentioned in the paper) from https://timeseriesclassification.com/ website, along with setting up the environment by install `Nengo`, `NengoLoihi`, and `PyTorch` libraries. 

To run the code in `energy-consumption-analysis`, you don't need to download datasets, but have access to Loihi-1 on INRC and of course an Intel CPU machine. You would still need to install `Nengo` and `NengoLoihi` libraries along with `pyJoules` library.
