# Mandatory Assignment 1

## Software Requirements
* Python 3.6 and newer
* All Python libraries are shown in requirements.txt: these are also the same which will be loaded upon "module load PyTorch-bundle/1.7.0" on ml6 or ml7 server.

## How to Run
By default, the code will
* use cuda (assuming that it is available). However, if GPU is not available, change the variable `config["use_gpu"]` to `False`.
* use shared data directory on ml7.hpc.uio.no server.

To run everything, execute:

```
$ python main.py
```

main.py simply controls the code which is in the /src directory as two python files task1.py and task2.py.