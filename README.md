# M-Track
Code for CVPR22 paper: [One Step at a Time: Long-Horizon Vision-and-Language Navigation with Milestones](https://arxiv.org/abs/2202.07028)



# Setup

1. Setup Conda Environment

```
conda create -n mtrack -f mtrack.yaml  
conda activate mtrack
```

2. Set local variables

```
export $MTRACK=$(pwd)
```

3. Download ALFRED

```
mkdir data
cd data
git clone https://github.com/askforalfred/alfred.git
export ALFRED_ROOT=$(pwd)/alfred
ln -s $ALFRED_ROOT/data/json_2.1.0 $MTRACK/data/
```

# Training

```
bash run/train_agent.bash
```

# Testing

Evaluation on Validation Unseen split

```
bash run/test_agent.bash
```


# Bibtex

```
@InProceedings{Song_2022_CVPR,
    author    = {Song, Chan Hee and Kil, Jihyung and Pan, Tai-Yu and Sadler, Brian M. and Chao, Wei-Lun and Su, Yu},
    title     = {One Step at a Time: Long-Horizon Vision-and-Language Navigation With Milestones},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {15482-15491}
}
```
