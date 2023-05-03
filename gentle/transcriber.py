import math
import logging
import wave
import os
import subprocess
from  itertools import repeat
from .util.paths import get_datadir
import uuid

from gentle import transcription

from multiprocessing.pool import ThreadPool as Pool

class MultiThreadedTranscriber:
    def __init__(self, hclg_path, resources, chunk_len=20, overlap_t=2, nthreads=4, lang='lang', kaldi_queue =None):
        self.hclg_path = hclg_path
        self.resources=resources
        self.chunk_len = chunk_len
        self.overlap_t = overlap_t
        self.nthreads = nthreads
        self.lang=lang
        self.kaldi_queue = kaldi_queue

    def transcribe(self, wavfile, wavfile_path, gpu_id, max_batch_size, cuda_memory_prop, minibatch_size,  device, output_folder, progress_cb=None):
        wav_obj = wave.open(wavfile, 'rb')
        duration = wav_obj.getnframes() / float(wav_obj.getframerate())
        n_chunks = int(math.ceil(duration / float(self.chunk_len - self.overlap_t)))

        chunks = []

        """
        This if else statement switches between cpu and gpu decoder.
        ## TODO: Refactor this switch with a proper pattern
        """
        if device ==  'gpu' :
            #GPU Decoder

            # 0 - create a folder in kaldi_decoding  for this decoding job
            audio_file_path_list = wavfile_path.split('/')
            audio_file_name = audio_file_path_list[len(audio_file_path_list)-1]
            job_folder_name = audio_file_name.split('.')[0] + "_Gentle_GPU_JOB_" + str(uuid.uuid4())

            lang_folder_name =  self.lang + "_exp"

            os.chdir(get_datadir('kaldi_decoding'))

            # 1 - create the job folder in the provided path. /tmp folder if the user doesn't specify any output.txt file
            try:
                os.mkdir(os.path.join(output_folder,job_folder_name))
            except FileExistsError:
                pass


            # 2 - create wav.scp file which has one single entry
            wav_scp_file = os.path.join(job_folder_name, 'wav.scp')
            wav_scp_path = os.path.join(output_folder, wav_scp_file)
            wav_scp =  open(wav_scp_path, 'w')



            """
            This function fills the wav.scp file. It needs to have a single line for each segment. Each of these segments will be decoded
            separatly by the cuda decoder.
            """
            def chunk_to_wavscp_segment(idx, chunks_number):
                # extract start time of each segment
                wav_obj = wave.open(wavfile, 'rb')
                #start time of this audio chunk
                start_t = idx * (self.chunk_len - self.overlap_t)

                duration = int(self.chunk_len * wav_obj.getframerate())
                duration_seconds = duration / wav_obj.getframerate()

                # add the chunk informations in a new line in the wav.scp file
                recording_id = "utt1"
                utterance_id = recording_id + "_"  + str(start_t) + "_" +  str(duration_seconds)
                line = utterance_id + " ffmpeg -vn -ss "+ str(start_t)  +" -t "+ str(duration_seconds)  + " -i "  + wavfile_path + " -ac 1 -ar 16000 -f wav -| \n"
                wav_scp.write(line)

            pool = Pool(min(n_chunks, 1))

            # 3 - Populate wav_scp file with informations regarding each chunk
            print("Creating Kaldi files for this audio file ")
            pool.starmap(chunk_to_wavscp_segment, zip(range(n_chunks),repeat(n_chunks)))
            pool.close()
            wav_scp.close()
            chunks.sort(key=lambda x: x['start'])


            # 4 - launch external bash script for kaldi decoding on gpu
            print("Launching kaldi to decode the input audio file")
            kaldi_process = subprocess.check_output(
                                ["./kaldi_decode.sh", self.lang, job_folder_name, gpu_id, self.hclg_path, str(max_batch_size), str(cuda_memory_prop), str(minibatch_size), self.resources.model_name, output_folder ])
            #kaldi_process = subprocess.run(
            #                    ["./kaldi_decode.sh", self.lang, job_folder_name, gpu_id, self.hclg_path, str(max_batch_size), str(cuda_memory_prop), str(minibatch_size), self.resources.model_name, output_folder ],
            #                    check=True)

            # 5 - populate the chunk string with the trascription and starting time of each segment (should retrive this information from decodings or lattices)
            print("Create Gentle data structures with decoded text from Kaldi ")
            language_folder= self.lang+"_exp"
            trascription_file = os.path.join(output_folder, job_folder_name, "transcript.txt")
            print(output_folder)
            print(job_folder_name)
            print(trascription_file)
            decoding_file = open(trascription_file, 'r')
            words_of_each_segment = decoding_file.readlines()


            # 6 - sort all the segments of the trascription in the proper order.
            def extract_starting_time(line):
                utt_id = line.split(" ")[0]
                start_t_string = utt_id.split("_")[1]
                start_t = int(start_t_string)
                return start_t

            words_of_each_segment.sort(key=extract_starting_time)

            #retrieve the informations from kaldi decoding file into the all_segments list
            all_segments = []
            for idx, line in enumerate(words_of_each_segment):
                utt_id = line.split(" ")[0]
                #only for the first line
                if idx == 0 :
                    segment = []
                    segment.append(line)
                #for all the others line
                else:
                    if utt_id == words_of_each_segment[idx-1].split(" ")[0]:
                        segment.append(line)
                        #only for the last line
                        if idx == len(words_of_each_segment) -1 :
                            all_segments.append(segment)
                    else:
                        #first save the previous segment list to the main list
                        all_segments.append(segment)
                        #then procede creating a new segment list
                        segment = []
                        segment.append(line)

            #Refactor the information into the proper data structure expected by Gentle
            for idx, segment in enumerate(all_segments):
                ret = []

                start_t_seg =  extract_starting_time(all_segments[idx][0])

                for word in segment:
                    start_t = float(word.split(" ")[2])
                    word_duration = float(word.split(" ")[3])
                    word = word.split(" ")[4]
                    wd = {
                        "word": word,
                        "start": start_t,
                        "duration": word_duration,
                        "phones": []
                    }
                    ret.append(wd)

                chunks.append({"start": start_t_seg , "words": ret})
        else:
            #CPU decoder

            def transcribe_chunk(idx):
                try:
                    wav_obj = wave.open(wavfile, 'rb')
                    start_t = idx * (self.chunk_len - self.overlap_t)
                    # Seek
                    wav_obj.setpos(int(start_t * wav_obj.getframerate()))
                    # Read frames
                    buf = wav_obj.readframes(int(self.chunk_len * wav_obj.getframerate()))

                    if len(buf) < 4000:
                        logging.info('Short segment - ignored %d' % (idx))
                        ret = []
                    else:
                        k = self.kaldi_queue.get()
                        k.push_chunk(buf)
                        ret = k.get_final()
                        # k.reset() (no longer needed)
                        self.kaldi_queue.put(k)

                    chunks.append({"start": start_t, "words": ret})
                    logging.info('%d/%d' % (len(chunks), n_chunks))
                    if progress_cb is not None:
                        progress_cb({"message": ' '.join([X['word'] for X in ret]),
                                     "percent": len(chunks) / float(n_chunks)})
                except Exception as e:
                    logging.error("Caught an exception {}".format(e))
                    ret = []

            pool = Pool(min(n_chunks, self.nthreads))
            pool.map(transcribe_chunk, range(n_chunks))
            pool.close()
            chunks.sort(key=lambda x: x['start'])



        # 7 - Once the Chunks list is populetd with the cpu or gpu decoder, continue with Gentle code

        # Combine chunks
        words = []
        for c in chunks:
            chunk_start = c['start']
            chunk_end = chunk_start + self.chunk_len

            chunk_words = [transcription.Word(**wd).shift(time=chunk_start) for wd in c['words']]

            # At chunk boundary cut points the audio often contains part of a
            # word, which can get erroneously identified as one or more different
            # in-vocabulary words.  So discard one or more words near the cut points
            # (they'll be covered by the ovlerap anyway).
            #
            trim = min(0.25 * self.overlap_t, 0.5)
            if c is not chunks[0]:
                while len(chunk_words) > 1:
                    chunk_words.pop(0)
                    if chunk_words[0].end > chunk_start + trim:
                        break
            if c is not chunks[-1]:
                while len(chunk_words) > 1:
                    chunk_words.pop()
                    if chunk_words[-1].start < chunk_end - trim:
                        break

            words.extend(chunk_words)

        # Remove overlap:  Sort by time, then filter out any Word entries in
        # the list that are adjacent to another entry corresponding to the same
        # word in the audio.
        words.sort(key=lambda word: word.start)
        words.append(transcription.Word(word="__dummy__"))
        words = [words[i] for i in range(len(words)-1) if not words[i].corresponds(words[i+1])]

        return words, duration


if __name__=='__main__':
    # full transcription
    import json
    import sys

    import logging
    logging.getLogger().setLevel('INFO')

    import gentle
    from gentle import standard_kaldi
    from gentle import kaldi_queue

    resources = gentle.Resources()

    k_queue = kaldi_queue.build(resources, 3)
    trans = MultiThreadedTranscriber(k_queue)

    with gentle.resampled(sys.argv[1]) as filename:
        words, duration = trans.transcribe(filename)

    open(sys.argv[2], 'w').write(transcription.Transcription(words=words).to_json())
