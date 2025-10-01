# DRAG: Data Reconstruction Attack using Guided Diffusion

![](https://badgen.net/github/license/ntuaislab/DRAG)
[![arXiv](https://img.shields.io/badge/arXiv-2509.11724-b31b1b.svg)](https://www.arxiv.org/abs/2509.11724)

## [ArXiv](https://www.arxiv.org/abs/2509.11724) | [Poster & Slides](https://icml.cc/virtual/2025/poster/43496)

## 🚀 Getting Started

### Environment

We provided dependencies in conda for reproducing.

```bash
conda create -f environment.yml
```

Extra dependencies: CUDA, which is for compiling plugin for StyleGAN2-ADA, which is not a necessary part for our diffusion based attacks.

Download necessary dataset and checkpoint, which are storing in `<PROJ_ROOT>/datasets` and `<PROJ_ROOT>/checkpoints` by default.
```bash
<PROJ_ROOT>
├── checkpoints
│   ├── stylegan2-ada-pytorch
│   │   └── ffhq.pkl
│   └── stylegan-xl
│       └── imagenet256.pkl
├── datasets
│   ├── coco
│   ├── ffhq
│   └── imagenet2012
└── ...
```

To reproduce the results in the paper, check commands in the script `run.sh`.

### Experiment Settings

We sample 10 images for each dataset, determined with the script `roll_dice.py` with random seed 0.

| Dataset | Samples                                                    |
| :------ | :--------------------------------------------------------- |
| MSCOCO  | 119,138,725,1044,1703,1919,2111,2591,4111,4497             |
| FFHQ    | 337,429,1729,1917,2890,4919,6044,7532,8223,9399            |
| IN-1K   | 6091,11341,16904,17849,24681,28026,36044,36293,37807,49165 |

## 🎯 Model Checkpoint

TO BE UPDATED

## 🤝 Acknowledgment

This work builds on and benefits from several open-source efforts:

- [SIMBA: Split Inference - Metrics, Benchmarks and Algorithms](https://github.com/aidecentralized/InferenceBenchmark)
- [Guidance with Spherical Gaussian Constraint for Conditional Generation](https://github.com/LingxiaoYang2023/DSG2024)
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 📄 Citation

If you find our work useful, please cite us:

```bibtex
@inproceedings{lei2025drag,
    title={DRAG: Data Reconstruction Attack with Guided Diffusion},
    author={Wa-Kin Lei and Jun-Cheng Chen and Shang-Tse Chen},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
}
```

