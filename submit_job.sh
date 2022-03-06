#!/bin/bash

array=(
  xlnet_lr5_lc_nr.sh
  xlnet_lr6_lc_nr.sh
  xlnet_lr5_lc_sr_stem.sh
  xlnet_lr6_lc_sr_stem.sh
  xlnet_lr5_lc_sr_lemma.sh
  xlnet_lr6_lc_sr_lemma.sh
  xlnet_lr5_nr_sr_stem.sh
  xlnet_lr6_nr_sr_stem.sh
  xlnet_lr5_nr_sr_lemma.sh
  xlnet_lr6_nr_sr_lemma.sh
  xlnet_lr5_lc_nr_sr_stem.sh
  xlnet_lr6_lc_nr_sr_stem.sh
  xlnet_lr5_lc_nr_sr_lemma.sh
  xlnet_lr6_lc_nr_sr_lemma.sh
)

for i in ${array[@]}; do
#for ((i=107;i<=115;i++)); do
  echo "${i}" ;
  sbatch ${i};
  sleep 1;
done