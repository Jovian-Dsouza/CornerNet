
# CornerNet Implementation in Pytorch-Lightning

CornerNet is a new approach to object detection where we detect an object bounding box as a pair of keypoints (top-left corner and bottom-right corner)

This repo contains implementation of CornerNet in Pytorch-Lightning 

[Paper](https://arxiv.org/abs/1808.01244)

## Getting started 

### Clone this Repo
```bash
CornerNet_ROOT=/path/to/clone/CornerNet
git clone https://github.com/Jovian-Dsouza/CornerNet.git $CornerNet_ROOT
```

### Install Corner Pooling

```bash
cd $CornerNet_ROOT/lib/cpool
python setup.py install --user
```

## References

[zzzxxxttt/pytorch_simple_CornerNet](https://github.com/zzzxxxttt/pytorch_simple_CornerNet)
[princeton-vl/CornerNet](https://github.com/princeton-vl/CornerNet)