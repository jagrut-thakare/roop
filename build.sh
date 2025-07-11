conda create -n roop python=3.9 -y
conda activate roop
conda install cudatoolkit=11.8 nccl cudnn
pip install -r requirements.txt
