# PATS: Patch Area Transportation with Subdivision for Local Feature Matching
### [Project Page](https://zju3dv.github.io/pats/) | [Paper](https://arxiv.org/pdf/2303.07700.pdf)
<br/>

> PATS: Patch Area Transportation with Subdivision for Local Feature Matching  
> [Junjie Ni*](https://github.com/xuanlanxingkongxia) [Yijin Li*](https://eugenelyj.github.io/), [Zhaoyang Huang](https://drinkingcoder.github.io), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao), [Zhaopeng Cui](https://zhpcui.github.io), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang)  
> CVPR 2023

![Demo Video](https://raw.githubusercontent.com/eugenelyj/open_access_assets/master/pats/201.gif)


## TODO List
- [ ] Training script


## Download Link

We provide the [download link](https://drive.google.com/drive/folders/1SEz5oXVH1MQ2Q9lzLmz_6qQUoe6TAJL_?usp=sharing) to
  - Pretrained models trained on MegaDepth and ScanNet, which are labeled as outdoor and indoor, respectively.
  - MegaDepth pairs and scenes (placed in a folder named megadepth_parameters).
  - The demo data, which is a sequence of images captured from near to far.


## Run PATS

### Installation
```bash
conda env create -f environment.yaml
cd setup
python setup.py install
cd ..
```


### Prepare the data and pretrained model
Download from the above link, and place the data and model weights as below: 


```
pats
├── data
│   ├── MegaDepth_v1
│   ├── megadepth_parameters 
│   ├── ScanNet
│   ├── yfcc100M
│   └── demo
└── weights
    ├── indoor_coarse.pt
    ├── indoor_fine.pt
    ├── indoor_third.pt
    ├── outdoor_coarse.pt
    ├── outdoor_fine.pt
    └── outdoor_third.pt
```

### Evaluate on MegaDepth/YFCC/ScanNet dataset

```bash
python evaluate.py configs/test_megadepth.yaml
python evaluate.py configs/test_yfcc.yaml
python evaluate.py configs/test_scannet.yaml
```

### Run the demo

```bash
python demo.py configs/test_demo.yaml
```


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{pats2023,
  title={PATS: Patch Area Transportation with Subdivision for Local Feature Matching},
  author={Junjie Ni, Yijin Li, Zhaoyang Huang, Hongsheng Li, Hujun Bao, Zhaopeng Cui, Guofeng Zhang},
  booktitle={The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2023}
}
```


## Acknowledgements

We would like to thank the authors of [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) and [LoFTR](https://github.com/zju3dv/LoFTR) for open-sourcing their projects.
