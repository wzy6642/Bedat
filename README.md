# Behavior Data Augmentation with Transformers

## How To Use

### Preprocessing
- Step1: Download the preprocessed data from https://drive.google.com/drive/folders/1kSbJHwWwz5zG7wlFCP88wtd-XK1wNpSo
- Put the data file in <tt> data/</tt>

### Training 
#### Example of training on Twitter dataset:
```python
python run_loop.py --mode=train --cluster_num=5 --num_epochs=5 --gpu_id=0 \ 
                   --model_dir=./ckpt --learning_rate=1e-4 --num_epochs=10 --pretrain_dir=./pretrain
```
