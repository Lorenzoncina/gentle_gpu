#! /usr/bin/python3

import argparse
import logging
import multiprocessing
import os
import sys
from gentle.util.paths import get_datadir

import gentle

parser = argparse.ArgumentParser(
        description='Align a transcript to audio by generating a new language model.  Outputs JSON')
parser.add_argument(
        '--nthreads', default=multiprocessing.cpu_count(), type=int,
        help='number of alignment threads')
parser.add_argument(
        '-o', '--output', metavar='output', type=str,
        help='output filename')
parser.add_argument(
        '--conservative', dest='conservative', action='store_true',
        help='conservative alignment')
parser.set_defaults(conservative=False)
parser.add_argument(
        '--disfluency', dest='disfluency', action='store_true',
        help='include disfluencies (uh, um) in alignment')
parser.set_defaults(disfluency=False)
parser.add_argument(
        '--log', default="INFO",
        help='the log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL)')
parser.add_argument(
        '--lang', default="en",
        help='language of alignment (en, en_gentle fr, es, ar, ru, zh)')
parser.add_argument(
        '--gpu_id', type=str, default="0",
        help='gpu id to use')
parser.add_argument(
        '--device', type=str, default="cpu",
        help='Decoder type between cpu and gpu one')
parser.add_argument(
        '--max_batch_size', type=int, default=128,
        help='The maximum batch size to be used by the decoder. This is also the number of lanes in the CudaDecoder. Larger = Faster and more GPU memory used')
parser.add_argument(
        '--cuda_memory_proportion', type=float, default=0.5,
        help='Proportion of the GPU device memory that the allocator should allocate at the start (float, default = 0.5)')
parser.add_argument(
        '--minibatch_size', type=int, default=128,
        help='Number of chunks per minibatch')
parser.add_argument(
        'audiofile', type=str,
        help='audio file')
parser.add_argument(
        'txtfile', type=str,
        help='transcript text file')


args = parser.parse_args()

log_level = args.log.upper()
logging.getLogger().setLevel(log_level)

lang = args.lang
gpu_id=args.gpu_id
decoder_type = args.device
max_batch_size = args.max_batch_size
cuda_memory_prop = args.cuda_memory_proportion
minibatch_size = args.minibatch_size

#extract the path to the folder where decoding logs will be generet
output_folder = os.path.dirname(args.output)
output_file_path = os.path.abspath(args.output)

disfluencies = set(['uh', 'um'])

def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


with open(args.txtfile, encoding="utf-8") as fh:
    transcript = fh.read()

resources = gentle.Resources(lang)
if lang == 'en_gentle':
    logging.info("converting audio to 8K sampled wav")
else:
    logging.info("converting audio to 16K sampled wav")

with gentle.resampled(args.audiofile, lang) as wavfile:
    logging.info("starting alignment")
    aligner = gentle.ForcedAligner(resources, transcript, device = decoder_type, nthreads=args.nthreads, disfluency=args.disfluency, conservative=args.conservative, disfluencies=disfluencies, lang=lang)
    result = aligner.transcribe(wavfile, args.audiofile, gpu_id=gpu_id, max_batch_size=max_batch_size, cuda_memory_prop=cuda_memory_prop, minibatch_size=minibatch_size, output_folder = output_folder, progress_cb=on_progress, logging=logging)

fh = open(output_file_path, 'w', encoding="utf-8") if args.output else sys.stdout
fh.write(result.to_json(indent=2))
if args.output:
    logging.info("output written to %s" % (args.output))
