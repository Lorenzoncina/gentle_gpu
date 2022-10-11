import logging
import os

from .util.paths import get_resource, ENV_VAR
from . import metasentence

class Resources():

    def __init__(self, lang):
        
         self.lang = lang

         if lang == "en":
             self.proto_langdir = get_resource('exp/es_exp')
             self.nnet_gpu_path = get_resource('exp/es_exp/tdnn_7b_chain_online/')
             self.full_hclg_path = get_resource('exp/es_exp/tdnn_7b_chain_online/graph_pp/HCLG.fst')
         elif lang == "fr":
             self.proto_langdir = get_resource('exp/fr_exp')
             self.nnet_gpu_path = get_resource('exp/fr_exp/tdnn_7b_chain_online/')
             self.full_hclg_path = get_resource('exp/fr_exp/tdnn_7b_chain_online/graph_pp/HCLG.fst')
         elif lang =="es":
             self.proto_langdir = get_resource('exp/es_exp')
             self.nnet_gpu_path = get_resource('exp/es_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/es_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
         elif lang == "ar":
             self.proto_langdir= get_resource('exp/ar_exp/langdir')
             self.nnet_gpu_path = get_resource('exp/ar_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/ar_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
         elif lang == "ru":
             self.proto_langdir= get_resource('exp/ru_exp/langdir')
             self.nnet_gpu_path = get_resource('exp/ru_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/ru_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
         elif lang == "zh":
             self.proto_langdir= get_resource('exp/zh_exp/langdir')
             self.nnet_gpu_path = get_resource('exp/zh_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/zh_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
         elif lang == "pt":
             self.proto_langdir= get_resource('exp/pt_exp/langdir')
             self.nnet_gpu_path = get_resource('exp/pt_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/pt_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
         elif lang == "it":
             self.proto_langdir= get_resource('exp/it_exp/langdir')
             self.nnet_gpu_path = get_resource('exp/it_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/it_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
         elif lang == "en_wipo":
             self.proto_langdir= get_resource('exp/en_exp/langdir')
             self.nnet_gpu_path = get_resource('exp/en_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/en_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
         else:
             raise RuntimeError("language  is not supported")




         def require_dir(path):
             if not os.path.isdir(path):
                 raise RuntimeError("No resource directory %s.  Check %s environment variable?" % (path, ENV_VAR))


         require_dir(self.proto_langdir)
         require_dir(self.nnet_gpu_path)

         with open(os.path.join(self.proto_langdir, "langdir", "words.txt")) as fh:
             self.vocab = metasentence.load_vocabulary(fh)


