#!/bin/bash

set -e

#This script perform kaldi decoding using  gpu cuda decoder (batched-wav-nnet3-cuda) to improve
#s2t decoding performances
source ./path.sh

echo "start decoding"
export train_cmd="run.pl"
export decode_cmd="run.pl --mem 2G"

#Create decoding graph and decode audio
lang=$1_exp
model_name=$8
export dir=../exp/$lang/$model_name
export graph_dir=$dir/graph_pp

gentle_dir=${PWD}/..
job_folder_name=$2
output_folder=$9
output_dir=$output_folder/$job_folder_name

#GPU decoding with cuda decoder
graphdir=$graph_dir
iter=final
model=$dir/$iter.mdl

acwt=1.0  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=10.0
cmd=run.pl
beam=30.0
frames_per_chunk=50
lattice_beam=8.0
max_active=7000


if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$output_dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$output_dir/lat.JOB.gz"
fi

mkdir -p $output_dir/log

ivector_conf_path="conf/"$lang"_ivectors_conf/ivector.conf"
#select the proper gpu
export CUDA_VISIBLE_DEVICES=$3


$cmd  $output_dir/log/batched-wav-nnet3-cuda2-batchsize2.log \
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
    --max-batch-size=$5 \
    --batch-drain-size=128 \
    --edge-minibatch-size=128 \
    --minibatch-size=$7 \
    --cuda-use-tensor-cores=true \
    --cuda-memory-proportion=$6  \
    --num-channels=-1 \
    --beam=$beam \
    --lattice-beam=$lattice_beam \
    --max-active=$max_active \
    --word-symbol-table=$graphdir/words.txt \
    $model \
    $4 \
    scp:$output_dir/wav.scp \
    $lat_wspecifier


#set --frame-shift accprding to the language
if [ "$lang" == "en_exp" ]
then
	frame_shift=0.03
elif [ $lang == "en_gentle_exp" ]
then
	frame_shift=0.03
elif [ "$lang" == "fr_exp" ]
then
	frame_shift=0.03
elif [ "$lang" == "es_exp" ]
then
	frame_shift=0.03
elif [ "$lang" == "ar_exp" ]
then
	frame_shift=0.03
elif [ "$lang" == "zh_exp" ]
then
	frame_shift=0.03
elif [ "$lang" == "ru_exp" ]
then
	frame_shift=0.03
fi

#decode lattices
#move to the decoding folder for this job
cd $output_dir
echo "actual folder (should be the decoding folder for this job)"
echo "Script executed from: ${PWD}"

lattice-1best --lm-scale=12 "ark:zcat lat.JOB.gz |" ark:- | lattice-align-words $gentle_dir/exp/$lang/langdir/phones/word_boundary.int $gentle_dir/exp/$lang/$model_name/final.mdl ark:- ark:- | nbest-to-ctm --frame-shift=0.03 ark:- - | $gentle_dir/kaldi_decoding/utils/int2sym.pl -f 5 $gentle_dir/exp/$lang/langdir/words.txt > transcript.txt

echo "end decoding"
