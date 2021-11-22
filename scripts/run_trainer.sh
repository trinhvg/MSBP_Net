#!/bin/bash
for network_name in ResNet_MSBP ResNet
do
  python trainer.py --dataset 'colon_tma' --seed 5 --gpu '0,1' --network_name ${network_name}
dones