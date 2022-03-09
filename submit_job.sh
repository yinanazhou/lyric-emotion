#!/bin/bash

array=(
  xlnet_lr5_lc_nr_sr.sh
  xlnet_lr6_lc_nr_sr.sh
)

for i in ${array[@]}; do
#for ((i=107;i<=115;i++)); do
  echo "${i}" ;
  sbatch ${i};
  sleep 1;
done
