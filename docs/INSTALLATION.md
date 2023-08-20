
# LFAI Documentation - Installation

This is the official documentation for the LFAI lstm model.


# Python Installation

### Windows 10/11
The first step is to install python if it is not already installed.
Go over to [Python](https://www.python.org/) and download the latest version.

### Linux/Unix
By default you will have python3 installed. Just check the version by typing the following command ``python3 --version`` and make sure the version is above ``3.6``

# Install Dependencies
Go into the folder which contains ``requirements.txt``. 

### Windows 10/11
Make sure to check if your computer has a [NVIDIA](https://nvidia.custhelp.com/app/answers/detail/a_id/5021/~/how-can-i-tell-what-graphics-card-i-have-in-my-computer%3F) gpu that supports [CUDA](https://developer.nvidia.com/cuda-downloads). If your computer does not support [CUDA](https://developer.nvidia.com/cuda-downloads) we will use the CPU version. 

You can execute the following command to install the requirements for an [NVIDIA](https://nvidia.custhelp.com/app/answers/detail/a_id/5021/~/how-can-i-tell-what-graphics-card-i-have-in-my-computer%3F) gpu.
```bash
python -m pip install -U -r requirements.txt
```
You can execute the following command to install the requirements for a CPU.
```bash
python -m pip install -U -r requirements_cpu.txt
```
### Linux/Unix
Check if your computer has a NVIDIA gpu by typing this command into the Terminal ``nvidia-smi``. If it returns ``Command 'nvidia-smi' not found`` you do not have an NVIDIA gpu installed or the drivers are not properly installed.

You can execute the following command to install the requirements for an [NVIDIA](https://developer.nvidia.com/nvidia-system-management-interface) gpu.
```bash
./install.sh
```
You can execute the following command to install the requirements for a CPU.
```bash
./install_cpu.sh
```

# Final Chapter

This is the end you did it!
Now you can move on to the [Next Section](/docs/TRAINING.md)!