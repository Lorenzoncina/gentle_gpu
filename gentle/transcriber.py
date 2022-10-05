import math
import logging
import wave
import os
import subprocess
from  itertools import repeat

from gentle import transcription

from multiprocessing.pool import ThreadPool as Pool

class MultiThreadedTranscriber:
    def __init__(self, kaldi_queue, chunk_len=20, overlap_t=2, nthreads=4):
        self.chunk_len = chunk_len
        self.overlap_t = overlap_t
        self.nthreads = nthreads

        self.kaldi_queue = kaldi_queue

    def transcribe(self, wavfile, wavfile_path,  progress_cb=None):
        wav_obj = wave.open(wavfile, 'rb')
        duration = wav_obj.getnframes() / float(wav_obj.getframerate())
        n_chunks = int(math.ceil(duration / float(self.chunk_len - self.overlap_t)))

        chunks = []

        # 0 -create a folder in kaldi/egs/wsj/data/ for this decoding job
        gentle_working_dir = os.getcwd()
        os.chdir('ext/kaldi/egs/wsj/s5/data')
        #create a new folder for this job where all kaldi files are then generated
        audio_file_path_list = wavfile_path.split('/')
        audio_file_name = audio_file_path_list[len(audio_file_path_list)-1]
        job_folder_name = audio_file_name.split('.')[0]+"_decoding_job"
        try:
            os.mkdir(job_folder_name)
        except FileExistsError:
            pass
    

        # 1 - create spk2utt and utt2spk files (in this case we don’t care about designating different speaker ids, so speaker id is equal to utterance id)
        utt2spk_file = os.path.join(job_folder_name, 'utt2spk')
        spk2utt_file = os.path.join(job_folder_name, 'spk2utt')
        utt2spk_path = os.path.join(os.getcwd(), utt2spk_file)
        spk2utt_path = os.path.join(os.getcwd(), spk2utt_file)
        utt2spk = open(utt2spk_path, 'w')
        spk2utt = open(spk2utt_path, 'w')


        # 2 - create wav.scp file which has one single entry
        wav_scp_file = os.path.join(job_folder_name, 'wav.scp')
        wav_scp_path = os.path.join(os.getcwd(), wav_scp_file)

        with open(wav_scp_path, 'w') as wav_scp:
            audio_file_path = gentle_working_dir +'/' + wavfile_path
            line = "utt1 " + " ffmpeg -vn -i "  + audio_file_path + " -ac 1 -ar 16000 -f wav -|"
            wav_scp.write(line)


        # 3 - create segment file, each entry will be one chunk of audio. it will be filled in the chunk_to_segment function
        seg_name = os.path.join(job_folder_name, 'segments')
        seg_file_path = os.path.join(os.getcwd(), seg_name)
        segments = open(seg_file_path, 'w')



        """
        This function fills the segments, utt2spk and spk2utt files with all the informations for each segment
        """
        def chunk_to_segment(idx, chunks_number):
            # 0 - extract <segment-begin> <segment-end> for the chunk
            wav_obj = wave.open(wavfile, 'rb')
            #start time of this audio chunk
            start_t = idx * (self.chunk_len - self.overlap_t)
            end_t = 0
            if (idx < chunks_number -1):
                duration = int(self.chunk_len * wav_obj.getframerate())
                duration_seconds = duration / wav_obj.getframerate()
                end_t = start_t + duration_seconds
            else:
                #for the last chunck, end_t will be the lenght of the entire audio file
                frames = wav_obj.getnframes()
                rate = wav_obj.getframerate()
                duration = frames / float(rate)
                end_t = duration



            # 1 - add the chunk informations in a new line in the segment file
            recording_id = "utt1"
            utterance_id = recording_id + "_"  + str(start_t) + "_" +  str(end_t)
            line = utterance_id + " " + recording_id + " " + str(start_t) + " " +  str(end_t) + "\n"
            segments.write(line)

            # 2
            line_2 = utterance_id + " " + utterance_id + "\n"
            utt2spk.write(line_2)
            spk2utt.write(line_2)


        #this function with the new cuda version is not necessary anymore (delete it after tests)
        def transcribe_chunk(idx):
            wav_obj = wave.open(wavfile, 'rb')
            start_t = idx * (self.chunk_len - self.overlap_t)
            # Seek
            wav_obj.setpos(int(start_t * wav_obj.getframerate()))
            # Read frames
            duration = int(self.chunk_len * wav_obj.getframerate())
            n_seconds = duration / wav_obj.getframerate()
            buf = wav_obj.readframes(duration)

            if idx == 0:
                path_chunk_wav = "/home/lorenzoncina/Documents/WIPO/code/gentle_gpu/gentle/gentle/chunk_wavs/new.wav"
                wav_obj_2 = wave.open( path_chunk_wav , 'wb')
                wav_obj_2.setnchannels(1)
                wav_obj_2.setsampwidth(2)
                wav_obj_2.setframerate(wav_obj.getframerate())
                wav_obj_2.writeframesraw(buf)

            end_t = start_t + self.chunk_len
            print("chunk idx: ", idx, " start time: ", start_t, " Chunk lenght in second: ", duration_seconds, "  end time: ", end_t)


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


        pool = Pool(min(n_chunks, self.nthreads))
        # 4 - create kaldi files and populate segments_file with informations regarding each chunk
        print("Creating Kaldi files for this audio file ")
        pool.starmap(chunk_to_segment, zip(range(n_chunks),repeat(n_chunks)))
        pool.close()
        segments.close()
        utt2spk.close()
        spk2utt.close()
        chunks.sort(key=lambda x: x['start'])


        # 5 - launch external bash script for kaldi decoding on gpu
        os.chdir('..')
        print("Launching kaldi to decode the input audio file")
        subprocess.call(["./kaldi_decode.sh", job_folder_name])

        # 6 - populate the chunk string with the trascription and starting time of each segment (should retrive this information from decodings or lattices)
        print("Create Gentle data structures with decoded text from Kaldi ")
        hard_coded_path = 'exp/chain_cleaned/tdnn_1d_sp/french_short_speech_decoding_job/transcript.txt'
        path = os.path.join( os.getcwd(), hard_coded_path)
        decoding_file = open(path, 'r')
        words_of_each_segment = decoding_file.readlines()

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
            start_t_seg = idx * (self.chunk_len - self.overlap_t)
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
        
        # 7 - continue with Gentle code providing che new generated chunks list

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

