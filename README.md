# csu-csc510-1

AI Foundations

## Prerequisites
* Windows w/ WSL2
* Ubuntu 22.04 LTS (Windows App Store)

## Windows Setup

Ensure WSL2 is enabled.

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
```

Ensure GPU is available to WSL Ubuntu and Windows

```
nvidia-smi
```

Install os dependencies on WSL Ubuntu

```
make setup-os
```

Ensure cuda toolkit

```
nvcc --version
```

## Setup Ubuntu 22.04 WSL2

Install python depends: `make setup-python`

Install cuda toolkit: `make setup-cuda-toolkit`

Install cuDNN: `make setup-cudnn`

## USAGE

help: `make`

### Simple ANN

> Assumes you have met os level and global python (we are using ubuntu python 3.10)

Setup: `make setup-simple-ann`
Execute: `make simple-ann`

### Informed Search - Towers of Hanoi

Execute: `make towers-hanoi`

### Portfolio Project

Setup Portfolio Project: `make pp`

Execute Portfolio Project: `make pp`

