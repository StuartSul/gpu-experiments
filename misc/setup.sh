#########################
# Install miniconda
#########################

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all

conda create --name stuart python=3.12 -y
conda activate stuart
echo "conda activate stuart" >> ~/.bashrc


#########################
# Setup Python
#########################

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install pybind11 numpy


##########################
# Setup CUDA
##########################

echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc


##########################
# Setup ThunderKittens
##########################

git clone https://github.com/HazyResearch/ThunderKittens

PYTHON_VERSION=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INCLUDES=$(python3 -c "import sysconfig; print('-I', sysconfig.get_path('include'), sep='')")
PYTHON_LIBDIR=$(python3 -c "import sysconfig; print('-L', sysconfig.get_config_var('LIBDIR'), sep='')")
PYBIND_INCLUDES=$(python3 -m pybind11 --includes)
PYTORCH_INCLUDES=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTORCH_LIBDIR=$(python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")

echo "export PYTHON_VERSION=\"${PYTHON_VERSION}\"" >> ~/.bashrc
echo "export PYTHON_INCLUDES=\"${PYTHON_INCLUDES}\"" >> ~/.bashrc
echo "export PYTHON_LIBDIR=\"${PYTHON_LIBDIR}\"" >> ~/.bashrc
echo "export PYBIND_INCLUDES=\"${PYBIND_INCLUDES}\"" >> ~/.bashrc
echo "export PYTORCH_INCLUDES=\"${PYTORCH_INCLUDES}\"" >> ~/.bashrc
echo "export PYTORCH_LIBDIR=\"${PYTORCH_LIBDIR}\"" >> ~/.bashrc


###################
# Set git config
###################

git config --global user.name "Stuart Sul" && git config --global user.email "ssstuartss@gmail.com"


###################
# Install Claude
###################

curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc


###################
# Install Codex
###################

npm i -g @openai/codex


###################
# Complete setup
###################

source ~/.bashrc


###################
# Test Single GPU
###################
cd ~/ThunderKittens/kernels/gemm/bf16_h100
make run


###################
# Test Multi GPU
###################
export GPU=H100
cd ~/ThunderKittens/kernels/parallel/ag_gemm
make run
