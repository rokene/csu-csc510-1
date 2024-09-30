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

Setup Portfolio Project: `make pp`

Execute Portfolio Project: `make pp`

