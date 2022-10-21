import logging
import os

from .util.paths import get_resource, ENV_VAR
from . import metasentence

class Resources():

    def __init__(self, lang):
        
         self.lang = lang

         if lang == "en":
             self.proto_langdir = get_resource('exp/en_exp')
             self.nnet_gpu_path = get_resource('exp/en_exp/tdnn_7b_chain_online/')
             self.full_hclg_path = get_resource('exp/en_exp/tdnn_7b_chain_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn_7b_chain_online'
         elif lang == "en_gentle":
             self.proto_langdir = get_resource('exp/en_gentle_exp')
             self.nnet_gpu_path = get_resource('exp/en_gentle_exp/tdnn_7b_chain_online/')
             self.full_hclg_path = get_resource('exp/en_gentle_exp/tdnn_7b_chain_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn_7b_chain_online'
         elif lang == "fr":
             self.proto_langdir = get_resource('exp/fr_exp')
             self.nnet_gpu_path = get_resource('exp/fr_exp/tdnn_7b_chain_online/')
             self.full_hclg_path = get_resource('exp/fr_exp/tdnn_7b_chain_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn_7b_chain_online'
         elif lang == "fr_lig":
             self.proto_langdir = get_resource('exp/fr_lig_exp')
             self.nnet_gpu_path = get_resource('exp/fr_lig_exp/tdnn_7b_chain_online/')
             self.full_hclg_path = get_resource('exp/fr_lig_exp/tdnn_7b_chain_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn_7b_chain_online'
         elif lang =="es":
             self.proto_langdir = get_resource('exp/es_exp')
             self.nnet_gpu_path = get_resource('exp/es_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/es_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn1a_sp_online'
         elif lang == "ar":
             self.proto_langdir= get_resource('exp/ar_exp')
             self.nnet_gpu_path = get_resource('exp/ar_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/ar_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn1a_sp_online'
         elif lang == "ru":
             self.proto_langdir= get_resource('exp/ru_exp')
             self.nnet_gpu_path = get_resource('exp/ru_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/ru_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn1a_sp_online'
         elif lang == "zh":
             self.proto_langdir= get_resource('exp/zh_exp')
             self.nnet_gpu_path = get_resource('exp/zh_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/zh_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn1a_sp_online'
         elif lang == "pt":
             self.proto_langdir= get_resource('exp/pt_exp')
             self.nnet_gpu_path = get_resource('exp/pt_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/pt_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn1a_sp_online'
         elif lang == "it":
             self.proto_langdir= get_resource('exp/it_exp')
             self.nnet_gpu_path = get_resource('exp/it_exp/tdnn1a_sp_online/')
             self.full_hclg_path = get_resource('exp/it_exp/tdnn1a_sp_online/graph_pp/HCLG.fst')
             self.model_name = 'tdnn1a_sp_online'
         else:
             raise RuntimeError("language  is not supported")




         def require_dir(path):
             if not os.path.isdir(path):
                 raise RuntimeError("No resource directory %s.  Check %s environment variable?" % (path, ENV_VAR))


         require_dir(self.proto_langdir)
         require_dir(self.nnet_gpu_path)

         with open(os.path.join(self.proto_langdir, "langdir", "words.txt")) as fh:
             self.vocab = metasentence.load_vocabulary(fh)


