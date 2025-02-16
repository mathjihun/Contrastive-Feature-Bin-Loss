# Contrastive Feature Bin Loss for Monocular Depth Estimation
This repository contains the official implementation of CFBLoss, as presented in our paper:


## Usage
We set all settings to be identical to those of the corresponding model. Please refer to each model's repository for the environment required to use it.

If you want to use CFBLoss, then change the output of models from
```sh
return x
```
to
```sh
def BCP(...):  # make bin_centers
    ....


if self.training:
    return x, bin_centers, feature
else:
    return x
```

Additionally, add the CFB Loss with SILog loss. For example, change
```sh
output = model(input, target)
```
to
```sh
output, feature = model(input, target)
cfbloss = CFBLoss(feature, bin_centers, target)
loss = silog + cfbloss * cfbloss_weights
```

The code modifications for each model are as follows.

**PixelFormer**

./pixelformer/train.py
./pixelformer/networks/PixelFormer.py

**MIM-Depth-Estimation**

./models/model.py
./train.py

**Depth Anything**

./metric_depth/zoedepth/models/zoedepth/zoedepth_v1.py
./metric_depth/zoedepth/trainers/zoedepth_trainer.py
./metric_depth/zoedepth/models/base_models/dpt_dinov2/dpt.py



## Acknowledgements
This project was developed using resources from the following repositories:  

- [**PixelFormer**](https://github.com/ashutosh1807/PixelFormer)  
- [**MIM-Depth-Estimation**](https://github.com/SwinTransformer/MIM-Depth-Estimation)  
- [**Depth Anything**](https://github.com/LiheYoung/Depth-Anything)  
- [**OrdinalEntropy**](https://github.com/needylove/OrdinalEntropy)  

We sincerely appreciate the contributions of the respective authors and acknowledge their valuable work. 
 
