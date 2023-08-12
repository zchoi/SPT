<div align="center">
<h1>
<b>
SPT: Spatial Pyramid Transformer for Image Captioning
</b>
</h1>
<h4>
<a href="https://github.com/zchoi">Haonan Zhang</a>, <a href="https://ppengzeng.github.io/">Pengpeng Zeng</a>, <a href="https://lianligao.github.io/">Lianli Gao</a>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=kVcO9R4AAAAJ&view_op=list_works&sortby=pubdate">Xinyu Lyu</a>, <a href="https://cfm.uestc.edu.cn/~songjingkuan/">Jingkuan Song</a>, <a href="https://cfm.uestc.edu.cn/~shenht/">Heng Tao Shen</a>, 
</h4>

[Paper] | **TCSVT23 Submission** 
</div>
This is the code implementation of the submitted paper "SPT: Spatial Pyramid Transformer for Image Captioning", the checkpoint and feature will be released soon.

## Overview 
The canonical approaches to image captioning tend to vision transformers to learn sentence generation. These methods typically treat visually representative modeling of an image as a sequential problem (*i.e.*, flatting image patches), which demonstrates impressive levels of performance. However, the spatial semantic loss for flattened grid features of images has not received much attention to date. Besides, the routine of the current transformer models tend to maintain a full-length patch sequence during training and inference, which lacks hierarchal representation and makes it difficult to generate sentences with multiple levels of granularity. To this end, we propose a Spatially Pyramidal Transformer (SPT), which progressively pools vision patches to shrink sequence length for caption generation with varying graininess among image grids.

<p align="center">
    <img src=framework.png><br>
    <span><b>Figure 1. Overview of the Spatial Pyramid Transformer (SPT) for Image Captioning.</b></span>
</p>

<!-- ## The Proposed Modules in SPT

- Spatial-aware Pseudo-supervised (SP) —— solving spatial information loss of grid caused by flatten operation.

- Scale-aware Reinforcement (SR) ——simultaneously explore both low- and high-level semantics.
 
<p align="center">
    <img src=imgs/SP.png  width="56%">
    <img src=imgs/SR.png  width="40%" height="20%"> <br>
    <span><b>Figure 2. Spatial-aware Pseudo-supervised. Right: Scale-aware Reinforcement.</b></span>
</p> -->

## Dataset and Training Details 
> Note: For the data preparation, feature download, and training details, please refer to this [Repo](https://github.com/zchoi/S2-Transformer).
