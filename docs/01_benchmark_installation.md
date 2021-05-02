# Benchmark installation



<br>

## 1. Setup Conda

```
# Conda installation

# For Linux
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# For OSX
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

chmod +x ~/miniconda.sh    
./miniconda.sh  

source ~/.bashrc          # For Linux
source ~/.bash_profile    # For OSX
```


<br>

## 2. Setup Python environment for CPU

```
# Clone GitHub repo
conda install git
git clone https://github.com/graphdeeplearning/benchmarking-gnns.git
cd benchmarking-gnns

# Install python environment
conda env create -f environment_cpu.yml   

# Activate environment
conda activate benchmark_gnn
```



<br>

## 3. Setup Python environment for GPU

DGL 0.6.1 requires CUDA **10.2**.

For Ubuntu **18.04**

```
# Setup CUDA 10.2 on Ubuntu 18.04
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo apt --purge remove "nvidia*"
sudo apt autoremove
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.2.89-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install -y cuda-10-2
sudo reboot
cat /usr/local/cuda/version.txt # Check CUDA version is 10.2

# Clone GitHub repo
conda install git
git clone https://github.com/graphdeeplearning/benchmarking-gnns.git
cd benchmarking-gnns

# Install python environment
conda env create -f environment_gpu.yml 

# Activate environment
conda activate benchmark_gnn
```






<br><br><br>

