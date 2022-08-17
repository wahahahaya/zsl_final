please download the dataset https://drive.google.com/file/d/1qmMc7u2xMUuAomNRJwvoTR3DMGyO6YOu/view?usp=sharing  
and download the pre-feature from https://drive.google.com/file/d/18zG_BZjW2W9yOKmsvV593ZxXqtU1ox6p/view?usp=sharing

CUB:
```
python CE_GZSL.py --dataset CUB --class_embedding sent --syn_num 300 --batch_size 2048 --attSize 1024 --nz 1024 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --ins_temp 0.1 --cls_temp 0.1 --manualSeed 3483 --nclass_all 200 --nclass_seen 150 --nepoch 1000
```

SUN:
```
python CE_GZSL.py --dataset SUN --class_embedding att --syn_num 100 --batch_size 1024 --attSize 102 --nz 102 --embedSize 2048 --outzSize 512 --nhF 1024 --ins_weight 0.01 --cls_weight 0.01 --ins_temp 0.1 --cls_temp 0.1 --manualSeed 4115 --nclass_all 717 --nclass_seen 645 --lr_decay_epoch 100 --lr 5e-5 --nepoch 350
```

AWA2 with H=68.4:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 1.0 --ins_temp 10.0 --manualSeed 9182 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 130
```

###############  
AWA2 with randomseed:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 1.0 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with randomseed:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 1.0 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with randomseed:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 1.0 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with randomseed:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 1.0 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with seed 9182 syn_num 1000:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 1000 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 1.0 --ins_temp 10.0 --manualSeed 9182 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with seed 9182 syn_num 3000:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 3000 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 1.0 --ins_temp 10.0 --manualSeed 9182 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with seed 9182 syn_num 2000:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2000 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 1.0 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with seed 9182 cls_temp 2.0:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 2.0 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with seed 9182 cls_temp 3.0:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 3.0 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with seed 9182 cls_temp 0.1:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 0.1 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```

AWA2 with seed 9182 cls_temp 0.5:
```
python CE_GZSL.py --dataset AWA2 --class_embedding att --syn_num 2400 --batch_size 4096 --attSize 85 --nz 85 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 --cls_weight 0.001 --cls_temp 0.5 --ins_temp 10.0 --manualSeed 0 --lr_decay_epoch 10 --nclass_all 50 --nclass_seen 40 --lr 1e-4 --nepoch 200
```
