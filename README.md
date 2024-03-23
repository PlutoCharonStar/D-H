### Prepare environment
```bash
conda create -n test python=3.8
conda activate test
pip install -r requirements.txt
```

### Prepare test dataset
put 854 test videos in the data/test 
### Download Pretrain model
download the "swin_tiny_patch244_window877_kinetics400_1k.pth",put the pth file in pretrained_weights/

### Test

down the trained model "swin_head_val-ltest_s_finetuned.pth",put the pth file in the checkpoint/

```bash
nohup python -u test.py  --o config/kwai_simpleVQA_test.yml --gpu_id 0 > log/kwai_simpleVQA_test.log 2>&1 &
```
or 
```bash
bash scripts/test.sh
```

the predition results will be saved in result/output.txt