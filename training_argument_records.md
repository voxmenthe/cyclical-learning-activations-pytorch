# Training Instructions for 



## Learning rate adjustments
Several learning rate arguments are supported:
--lr
--lr-stages 20 --lr-increment 2
--lr-=dfsdf
* I will add support for PyTorch's built-in learning rate optimization functions when I get around to it.


## Start training:
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --model 'DPN' --model-config DPN92 --threads 8 --lr 0.11 --lr-decay 0.93 --epochs 50

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --model 'DPN92()' --threads 8 --lr 0.11 --lr-decay 0.93 --epochs 50
## Weirdly never gets past 10% accuracy... don't know what's wrong!

#### Set to PreActResNet101(): 
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --threads 8 --epochs 100 --lr 0.08 --lr-decay -0.98



Resume the training with `python main.py --resume --lr=0.01`


#### Set to DPN92_reswgt(residual_weight=1.65, activation=eswish)
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --threads 8 --epochs 400 --lr 0.2 --lr-stages 30 --lr-increment 2

### test run
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --threads 16 --epochs 3 --lr 0.3 --lr-stages 3

CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 80 --lr 0.3 --lr-stages 5 --lr-increment 2 --save-checkpoints --checkpoint-accuracy-threshold 85 --min-checkpoint-improvement 0.21

### TESTING THE MULTI-MODEL COMPARISON CODE:
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 30 --lr 0.16
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 60 --lr 0.3 --lr-stages 3 --lr-increment 2
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 200 --lr 0.3 --lr-stages 10 --lr-increment 2
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 80 --lr 0.3 --lr-stages 5 --lr-increment 2 --save-checkpoints --checkpoint-accuracy-threshold 88 --min-checkpoint-improvement 0.21
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 3 --lr 0.3 --lr-stages 3 --lr-increment 3

### run but interrupted
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 160 --lr 0.25 --lr-stages 12 --lr-increment 2 --save-checkpoints --checkpoint-accuracy-threshold 88 --min-checkpoint-improvement 0.41
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 20 --lr 0.4 --lr-stages 10 --lr-increment 2 --save-checkpoints --checkpoint-accuracy-threshold 75.5 --min-checkpoint-improvement 0.31

CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 36 --lr 0.4 --lr-stages 11 --lr-increment 2 --save-checkpoints --checkpoint-accuracy-threshold 82.85 --min-checkpoint-improvement 0.31

CUDA_VISIBLE_DEVICES=0,1 python3 main_multi_tmp_1.py --threads 8 --epochs 6 --lr 0.4 --lr-stages 3 --lr-increment 2


CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 240 --lr 0.4 --lr-stages 12 --lr-increment 2 --save-checkpoints --checkpoint-accuracy-threshold 88.5 --min-checkpoint-improvement 0.31

CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 18 --epochs 12 --lr 0.4 --lr-stages 4 --lr-increment 2
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 18 --epochs 12 --lr 0.4 --lr-stages 4 --lr-increment 2 --save-checkpoints --checkpoint-accuracy-threshold 83.5

CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 280 --lr 0.4 --lr-stages 28 --lr-increment 1.5 --save-checkpoints --checkpoint-accuracy-threshold 91.6 --min-checkpoint-improvement 0.35


CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 60 --lr 0.4 --lr-stages 20 --lr-increment 1.5

### pure test run
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 6 --lr 0.6 --lr-stages 6 --lr-increment 1.5 --train-batch-size 60 --test-batch-size 30

### current
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 20 --lr 0.4 --lr-stages 5 --lr-increment 1.5

### next
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 20 --lr 0.6 --lr-stages 4 --lr-increment 2

### march 30 2018
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 3 --lr 0.34 --lr-stages 3 --lr-increment 5 --save-checkpoints --checkpoint-accuracy-threshold 95 --min-checkpoint-improvement 0.21

### march 31 2018
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 16 --epochs 5 --lr 0.34 --decreasing-cycle-lr --save-checkpoints --checkpoint-accuracy-threshold 95 --min-checkpoint-improvement 0.21

CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 18 --epochs 41 --lr 0.28 --lr-change-n-epoch 3 --lr-change-increment 0.25 --lr-trend-adjustment-factor 0.98 --decreasing-cycle-lr --save-checkpoints --checkpoint-accuracy-threshold 95 --min-checkpoint-improvement 0.21

April 1 2018
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 18 --epochs 41 --lr 0.28 --lr-change-n-epoch 3 --decreasing-cycle-lr --lr-change-increment 0.25 --lr-trend-adjustment-factor 0.98 --save-checkpoints --checkpoint-accuracy-threshold 95 --min-checkpoint-improvement 0.21

CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 18 --epochs 21 --lr-change-n-epoch 2 --use-cycle-lr-list --lr-cycle-list-repeat --save-checkpoints --checkpoint-accuracy-threshold 95 --min-checkpoint-improvement 0.21

April 6 2018
CUDA_VISIBLE_DEVICES=0,1 python3 main_multi.py --threads 18 --epochs 41 --inbatch-lr-decay --save-checkpoints --checkpoint-accuracy-threshold 95 --min-checkpoint-improvement 0.21