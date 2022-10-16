#!/bin/bash

#This script perform kaldi decoding using  gpu cuda decoder (batched-wav-nnet3-cuda) to improve
#s2t decoding performances
echo "start decoding"
export train_cmd="run.pl"
export decode_cmd="run.pl --mem 2G"

#Create decoding graph and decode audio
lang=$1_exp
export dir=../exp/$lang/tdnn_7b_chain_online
export graph_dir=$dir/graph_pp


#GPU decoding with cuda decoder
graphdir=$graph_dir
data=data/$2/
dir=$dir/$2
job_folder_name=$2
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
iter=final
model=$srcdir/$iter.mdl

acwt=1.0  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=10.0
cmd=run.pl
beam=15.0
frames_per_chunk=50
lattice_beam=8.0
max_active=7000


if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

mkdir -p $dir/log

stage=1
if [ $stage -le 1  ]; then
  ivector_conf_path="conf/"$job_folder_name"_ivectors_conf/ivector.conf"
  #select the proper gpu
  export CUDA_VISIBLE_DEVICE=$3

  $cmd  $dir/log/batched-wav-nnet3-cuda2-batchsize2.log \
    batched-wav-nnet3-cuda \
    --cuda-use-tensor-cores=false \
    --iterations=1 \
    --acoustic-scale=$acwt \
    --feature-type=mfcc \
    --mfcc-config=conf/mfcc_hires.conf \
    --ivector-extraction-config=$ivector_conf_path \
    --add-pitch=false \
    --online-pitch-config=conf/online_pitch.conf \
    --frame-subsampling-factor=3 \
    --frames-per-chunk=$frames_per_chunk \
    --global-cmvn-stats=data/french_test_big/cmvn.scp   \
    --gpu-feature-extract=false \
    --max-batch-size=128 \
    --num-channels=496 \
    --beam=$beam \
    --lattice-beam=$lattice_beam \
    --max-active=$max_active \
    --word-symbol-table=$graphdir/words.txt \
    $model \
    $4 \
    scp:$data/wav.scp \
    $lat_wspecifier
fi


#decode lattices
#move to the decoding folder for this job
cd ../exp/$lang/tdnn_7b_chain_online
cd $2
echo "actual folder (should be the decoding folder for this job)"
echo "Script executed from: ${PWD}"
#source ./path.sh

lattice-1best --lm-scale=12 "ark:zcat lat.JOB.gz |" ark:- | lattice-align-words ../../../../exp/$lang/langdir/phones/word_boundary.int ../final.mdl ark:- ark:- | nbest-to-ctm ark:- - | ../../../../kaldi_decoding/utils/int2sym.pl -f 5 ../../../../exp/$lang/langdir/words.txt > transcript.txt


echo "end decoding"

