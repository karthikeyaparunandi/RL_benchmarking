## UNDER CONSTRUCTION
Repository to benchmark our algorithms (D2C and D2C-2.0) with the state-of-the-art RL algorithms.



Instructions:

1) Install Anaconda (https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh)
2) Create a new conda environment - "conda create -n spinningup python=3.6"
3) Activate the conda environment after navigating to the ""RL_benchmarking" folder: "conda activate venv_conda".
4) Install spinning up : Navigate to libraries/spinningup and run - "pip install -e ." (add "sudo" if you must).
5) Install gym : Navigate to libraries/gym and run - "pip3 install -e ." (add "sudo" if you must).
6) Make sure MuJoCo-2.0 is installed properly - 
	i) Navigate to '.mujoco/' and see if it is named as 'mujoco200' (by default, it searches in this folder).
	ii) Add it to the path "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/karthikeya/.mujoco/mujoco200/bin"
7) After ensuring that MuJoCo2.0 is installed properly, install mujoco_py : "pip install -U 'mujoco-py<2.1,>=2.0"

Libraries used :

- mujoco_py for MuJoCo-2.0
- Anaconda3-5.3.0 
- tensorflow (Anaconda automatically installs tensorflow-1.14.0)
- OpenMPI

Prerequisites: 

- mujoco_py
- numpy-1.14.5

