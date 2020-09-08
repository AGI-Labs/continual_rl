# IMPALA installation instructions

IMPALA is based on https://github.com/facebookresearch/torchbeast

To install torchbeast such that it is accessible from continual_rl, do:
```
git clone https://github.com/facebookresearch/torchbeast.git
cd torchbeast
pip install -r requirements.txt
```

Then add the torchbeast path to your PYTHONPATH. For instance for me:
```
export PYTHONPATH=$PYTHONPATH:/home/snpowers/Git/torchbeast
```