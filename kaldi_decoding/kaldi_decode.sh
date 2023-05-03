#!/bin/bash

set -exv
set -o pipefail

running_file_name=$(basename "$0")

die() {
  echo "Exiting $running_file_name with the value $1 !!"
  exit "$1"
}

trap 'die $?' EXIT

#This script perform kaldi decoding using  gpu cuda decoder (batched-wav-nnet3-cuda) to improve
#s2t decoding performances

source ./path.sh
echo "start decoding"
#Create decoding graph and decode audio

lang=$1_exp
job_folder_name=$2
gpu_id=$3
fst_in=$4
max_batch_size=$5
cuda_memory_protection=$6
minibatch_size=$7
model_name=$8
output_folder=$9
dir=../exp/$lang/$model_name
graph_dir=$dir/graph_pp
output_dir=$output_folder/$job_folder_name

#GPU decoding with cuda decoder
graphdir=$graph_dir
iter=final
model=$dir/$iter.mdl
#gentle_dir=/data/s2t/gentle/
gentle_dir=${PWD}/..
acwt=1.0  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=10.0
cmd=run.pl
beam=30.0
frames_per_chunk=50
lattice_beam=8.0
max_active=7000
frame_subsampling_factor=3
batch_drain_size=128
edge_minibatch_size=128
num_channels=-1
frame_subsampling_factor=3
iterations=1

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$output_dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$output_dir/lat.JOB.gz"
fi

mkdir -p $output_dir/log
#ivector_conf_path="${gentle_dir}/kaldi_decoding/conf/${lang}_ivectors_conf/ivector.conf"
ivector_conf_path="conf/"$lang"_ivectors_conf/ivector.conf"
#select the proper gpu
export CUDA_VISIBLE_DEVICES=$gpu_id

$cmd $output_dir/log/batched-wav-nnet3-cuda2-batchsize2.log batched-wav-nnet3-cuda \
    --cuda-use-tensor-cores=false \
    --iterations=$iterations \
    --acoustic-scale=$acwt \
    --feature-type=mfcc \
    --mfcc-config=conf/mfcc_hires.conf \
    --ivector-extraction-config=$ivector_conf_path \
    --add-pitch=false \
    --online-pitch-config=conf/online_pitch.conf \
    --frame-subsampling-factor=$frame_subsampling_factor \
    --frames-per-chunk=$frames_per_chunk \
    --global-cmvn-stats=data/french_test_big/cmvn.scp   \
    --gpu-feature-extract=false \
    --max-batch-size=$max_batch_size \
    --batch-drain-size=$batch_drain_size \
    --edge-minibatch-size=$edge_minibatch_size \
    --minibatch-size=$minibatch_size \
    --cuda-use-tensor-cores=true \
    --cuda-memory-proportion=$cuda_memory_protection  \
    --num-channels=$num_channels \
    --beam=$beam \
    --lattice-beam=$lattice_beam \
    --max-active=$max_active \
    --word-symbol-table=$graphdir/words.txt \
    $model \
    $fst_in \
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
