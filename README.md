# RWCE
Conformal prediction (CP) is a general framework to quantify the predictive uncertainty of machine learning models that uses a set prediction to include the true label with a valid probability. To align the uncertainty measured by CP, conformal training methods minimize the size of the prediction sets. A typical way is to use a surrogate indicator function, usually Sigmoid or Gaussian error function. However, these surrogate functions do not have a uniform error bound to the indicator function, leading to uncontrollable learning bounds. In this paper, we propose a simple cost-sensitive conformal training algorithm that does not rely on the indicator approximation mechanism. Specifically, we theoretically show that minimizing the expected size of prediction sets is upper bounded by the expected rank of true labels. To this end, we develop an importance weighting strategy that assigns the weight using the rank of true label on each data. Our analysis provably demonstrates the tightness between the proposed weighted objective and the expected size of conformal prediction sets. Extensive experiments verify the validity of our theoretical insights, and superior empirical performance over other conformal training in terms of predictive efficiency with 21.38% reduction for average prediction set size.


## Running instructions

Please run the commands mentioned below to produce results:

# CIFAR100
1. Please remove `srun` from the beginning of commands if you are using lab gpus.

  
## 0. Train the Base Models only using CE loss
**DenseNet**
```
srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --train_rule None --baseloss CE --method Baseloss --arc densenet100 --cal_test_CP_score HPS
```
**ResNet**
```
srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar.py --batch_size 128 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --train_rule None --baseloss CE --method Baseloss --arc resnet110 --cal_test_CP_score HPS
```

## 1. Commands for Ours Experiments
### HPS train, HPS cal-test
```
srun torchrun --standalone --nproc_per_node=1 ./train/train_cifar.py --batch_size 128 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 60 --finetune_lr 0.04 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --IWCE_rank_exponent 1.0 --finetune_weight_decay 0.0008 --mu 0.1 --train_rule None --baseloss CE --method IWCE_Loss --arc resnet110 --train_CP_score HPS --cal_test_CP_score HPS --train_T 1.5 --finetune_CE 1

srun torchrun --standalone --nproc_per_node=1 ./train/train_cifar.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 64 --finetune_epochs 60 --finetune_lr 0.025 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --IWCE_rank_exponent 1.0 --finetune_weight_decay 0.0004 --mu 0.1 --train_rule None --baseloss CE --method IWCE_Loss --arc densenet100 --train_CP_score HPS --cal_test_CP_score HPS --train_T 1.8 --finetune_CE 1

```
### HPS train, APS cal-test
```
srun torchrun --standalone --nproc_per_node=1 ./train/train_cifar.py --batch_size 128 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 60 --finetune_lr 0.04 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --IWCE_rank_exponent 1.0 --finetune_weight_decay 0.0008 --mu 0.1 --train_rule None --baseloss CE --method IWCE_Loss --arc resnet110 --train_CP_score HPS --cal_test_CP_score APS --train_T 1.5 --finetune_CE 1

srun torchrun --standalone --nproc_per_node=1 ./train/train_cifar.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 64 --finetune_epochs 60 --finetune_lr 0.025 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --IWCE_rank_exponent 1.0 --finetune_weight_decay 0.0004 --mu 0.1 --train_rule None --baseloss CE --method IWCE_Loss --arc densenet100 --train_CP_score HPS --cal_test_CP_score APS --train_T 1.8 --finetune_CE 1
```
### HPS train, RAPS cal-test
```
srun torchrun --standalone --nproc_per_node=1 ./train/train_cifar.py --batch_size 128 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 60 --finetune_lr 0.04 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --IWCE_rank_exponent 1.0 --finetune_weight_decay 0.0008 --mu 0.1 --train_rule None --baseloss CE --method IWCE_Loss --arc resnet110 --train_CP_score HPS --cal_test_CP_score RAPS --train_T 1.5 --finetune_CE 1

srun torchrun --standalone --nproc_per_node=1 ./train/train_cifar.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 64 --finetune_epochs 60 --finetune_lr 0.025 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --IWCE_rank_exponent 1.0 --finetune_weight_decay 0.0004 --mu 0.1 --train_rule None --baseloss CE --method IWCE_Loss --arc densenet100 --train_CP_score HPS --cal_test_CP_score RAPS --train_T 1.8 --finetune_CE 1

```
### HPS train, SAPS cal-test
```
srun torchrun --standalone --nproc_per_node=1 ./train/train_cifar.py --batch_size 128 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 60 --finetune_lr 0.04 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --IWCE_rank_exponent 1.0 --finetune_weight_decay 0.0008 --mu 0.1 --train_rule None --baseloss CE --method IWCE_Loss --arc resnet110 --train_CP_score HPS --cal_test_CP_score SAPS --train_T 1.5 --finetune_CE 1

srun torchrun --standalone --nproc_per_node=1 ./train/train_cifar.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 64 --finetune_epochs 60 --finetune_lr 0.025 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --IWCE_rank_exponent 1.0 --finetune_weight_decay 0.0004 --mu 0.1 --train_rule None --baseloss CE --method IWCE_Loss --arc densenet100 --train_CP_score HPS --cal_test_CP_score SAPS --train_T 1.8 --finetune_CE 1
