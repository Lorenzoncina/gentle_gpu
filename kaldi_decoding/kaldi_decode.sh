#!/bin/bash

#This script perform kaldi decoding using  gpu cuda decoder (batched-wav-nnet3-cuda) to improve
#s2t decoding performances
echo "start decoding"
export train_cmd="run.pl"
export decode_cmd="run.pl --mem 2G"

#Create decoding graph and decode audio
export dir=exp/chain_cleaned/tdnn_1d_sp
export graph_dir=$dir/graph_tgsmall
utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang_test $dir $graph_dir
#time bash steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 1 --cmd "$decode_cmd" \
     #--online-ivector-dir exp/nnet3/ivectors_$1 $graph_dir data/$1/ $dir/$1 $1

#Integration of the decode.sh script inside here

graphdir=$graph_dir
data=data/$1/
dir=$dir/$1
job_folder_name=$1
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
    $graphdir/HCLG.fst \
    scp:$data/wav.scp \
    $lat_wspecifier
fi


#decode lattices
echo "folder for the decoding of this job"
echo $1 #debug printing the decoding folder
#move to the decoding folder for this job
cd exp/chain_cleaned/tdnn_1d_sp
cd $1
echo "actual folder (should be the decoding folder for this job)"
echo "Script executed from: ${PWD}"
#source ./path.sh

lattice-1best --lm-scale=12 "ark:zcat lat.JOB.gz |" ark:- | lattice-align-words ../../../../data/lang_test/phones/word_boundary.int ../final.mdl ark:- ark:- | nbest-to-ctm ark:- - | ../../../../utils/int2sym.pl -f 5 ../../../../data/lang_test/words.txt > transcript.txt

echo "end decoding"

