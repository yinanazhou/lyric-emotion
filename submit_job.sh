#!/bin/bash

array=(
  xlnet_lr5_nr.sh
  xlnet_lr6_nr.sh
  xlnet_lr5_sr.sh
  xlnet_lr6_sr.sh
  xlnet_lr5_stem.sh
  xlnet_lr6_stem.sh
  xlnet_lr5_lemma.sh
  xlnet_lr6_lemma.sh
  xlnet_lr5_lc_sr.sh
  xlnet_lr6_lc_sr.sh
  xlnet_lr5_lc_stem.sh
  xlnet_lr6_lc_stem.sh
  xlnet_lr5_lc_lemma.sh
  xlnet_lr6_lc_lemma.sh
  xlnet_lr5_nr_sr.sh
  xlnet_lr6_nr_sr.sh
  xlnet_lr5_nr_stem.sh
  xlnet_lr6_nr_stem.sh
  xlnet_lr5_nr_lemma.sh
  xlnet_lr6_nr_lemma.sh
  xlnet_lr5_sr_stem.sh
  xlnet_lr6_sr_stem.sh
  xlnet_lr5_sr_lemma.sh
  xlnet_lr6_sr_lemma.sh
)

for i in ${array[@]}; do
#for ((i=107;i<=115;i++)); do
  echo "${i}" ;
  sbatch ${i};
  sleep 1;
done