您在运行`BraTS 2017`数据集**推理**的时候，请将[此处](https://github.com/Strivy-ZSY/litesegformer3d/blob/main/lsf3d/training/network_training/litesegformer3d_trainer_tumor.py)的`splits[self.fold]['val']`的值改为和[nnFormer](https://github.com/282857341/nnFormer/blob/main/nnformer/dataset_json/tumor_dataset.json)统一的数据。
具体代码如下：
```py
            splits[self.fold]['val'] = np.array([
       'BRATS_058', 'BRATS_059', 'BRATS_076', 'BRATS_077', 'BRATS_099',
       'BRATS_113', 'BRATS_114', 'BRATS_124', 'BRATS_139', 'BRATS_151',
       'BRATS_152', 'BRATS_157', 'BRATS_190', 'BRATS_240', 'BRATS_242',
       'BRATS_295', 'BRATS_305', 'BRATS_325', 'BRATS_331', 'BRATS_362',
       'BRATS_389', 'BRATS_425', 'BRATS_432', 'BRATS_450'])
```

因为`ACDC`和`Synapse`数据集默认和`nnFormer`一致，故无需修改。





When you run **inference** on the `BraTS 2017` dataset, please change [here](https://github.com/Strivy-ZSY/litesegformer3d/blob/main/lsf3d/training/network_training/litesegformer3d_trainer_tumor.py) the value of `splits[self.fold]['val']` to the same value as [nnFormer](https://github.com/282857341/nnFormer/blob/main/nnformer/dataset_json/tumor_dataset.json) uniformly.
The specific code is as follows:
```py
            splits[self.fold]['val'] = np.array([
       'BRATS_058', 'BRATS_059', 'BRATS_076', 'BRATS_077', 'BRATS_099',
       'BRATS_113', 'BRATS_114', 'BRATS_124', 'BRATS_139', 'BRATS_151',
       'BRATS_152', 'BRATS_157', 'BRATS_190', 'BRATS_240', 'BRATS_242',
       'BRATS_295', 'BRATS_305', 'BRATS_325', 'BRATS_331', 'BRATS_362',
       'BRATS_389', 'BRATS_425', 'BRATS_432', 'BRATS_450'])
```

Since the `ACDC` and `Synapse` datasets are consistent with `nnFormer` by default, no changes are required.
