# Install Directions for Benchmarks

## Atari
Installed by default; see [this link](https://github.com/mgbellemare/Arcade-Learning-Environment)
for more details.

## CHORES
1. Install ai2thor per [these instructions](https://github.com/allenai/ai2thor). 
Note: headless ai2thor (not requiring an X-server) coming soon.

2. Install [crl_alfred](https://github.com/etaoxing/crl_alfred/tree/develop), a fork from Alfred that defines the CHORES environment:
```
git clone git@github.com:etaoxing/crl_alfred.git --branch develop
cd crl_alfred
pip install -r alfred/requirements.txt
pip install -e .
```

3. Download CHORES trajectories (about 1 GB) into a directory of your choice:
```
wget -O cora_trajs.zip "https://onedrive.live.com/download?cid=601D311D0FC404D4&resid=601D311D0FC404D4%2155915&authkey=APSnA-AKY4Yw_vA"
unzip cora_trajs.zip
```

4. Set the ALFRED_DATA_DIR environment variable to your chosen directory:
```
export ALFRED_DATA_DIR=<download_path>/cora_trajs
```

## MiniHack
Install MiniHack per [these instructions](https://github.com/MiniHackPlanet/MiniHack)

If you do not have sudo access, the Nethack Learning Environment (on which MiniHack is based), 
can be installed via [these instructions](https://github.com/facebookresearch/nle/issues/246).


## Procgen
[Procgen](https://github.com/openai/procgen) can be installed using:
```
pip install procgen
```
