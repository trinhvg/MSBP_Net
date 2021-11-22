#!/bin/bash
for network_name in ResNet_MSBP ResNet
do
  python infer_produce_predict_map_wsi.py --gpu '0,1' --network_name ${network_name} --saved_path ''
done


ss