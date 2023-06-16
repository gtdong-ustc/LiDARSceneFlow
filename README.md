This is the code for "Exploiting Rigidity Constraints for LiDAR Scene Flow Estimation".

## Prerequisities
Our model is trained and tested under:
* Python 3.6.9
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 1.5)
* scipy
* tqdm
* sklearn
* numba
* cffi
* pypng
* pptk

Compile the furthest point sampling, grouping and gathering operation for PyTorch. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).

```shell
cd pointnet2
python setup.py install
cd ../
```

### Train
Set `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section. Then run
```bash
python train.py config_train.yaml
```

## Citation

If you use this code for your research, please cite our paper.

```
@inproceedings{dong2022exploiting,
  title={Exploiting rigidity constraints for lidar scene flow estimation},
  author={Dong, Guanting and Zhang, Yueyi and Li, Hanlin and Sun, Xiaoyan and Xiong, Zhiwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12776--12785},
  year={2022}
}
```


