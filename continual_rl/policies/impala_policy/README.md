# IMPALA installation instructions

IMPALA is based on https://github.com/facebookresearch/torchbeast

To install torchbeast such that it is accessible from continual_rl, do:
```
conda create -n torchbeast python=3.7
conda activate torchbeast
conda install pytorch -c pytorch
pip install -r requirements.txt
```

Then add the torchbeast path to your PYTHONPATH. For instance for me:
```
export PYTHONPATH=$PYTHONPATH:/home/snpowers/Git/torchbeast
```