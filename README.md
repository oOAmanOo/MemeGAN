MemeGAN
==================

### Train
```
python train.py --img-dir /path/to/rico --s2w-dir /path/to/screen2words -e 30 -b 32 -p 15
```
```
python train.py --img-dir /path/to/rico --s2w-dir /path/to/screen2words -e 30 -b 32 -p 15
```

### Evaluation
```
python eval.py -ckpt /path/to/checkpoint
```
### Generate Caption
```
python generater.py --img-dir /path/to/rico python train.py -ckpt /path/to/checkpoint --image_id 54137
```

Go to site packages to modify the code in
 swin transformer
```
[//]: # (Change the code in the following files)
[//]: # (Remove ".models" of timm.models.layers in model_parts.py)
[//]: # (Or place ".models" back to timm.models.layers in model_parts.py)
```

Go to site packages to modify the code in
 Gemini-torch
```
[//]: # (Change the code in the following files)
[//]: # (Replace 'AutoregressiveWrapper' class in zeta to 'AutoRegressiveWrapper' )
```