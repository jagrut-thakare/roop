conda create -n roop python=3.9 -y
conda run -n roop conda install cudatoolkit=11.8 nccl cudnn
conda run -n roop pip install -r requirements.txt
