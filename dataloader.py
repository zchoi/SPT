import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
import array
import math
import h5py
from config import Constants
from bisect import bisect_left
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import threading

class BigFile:
    
    def __init__(self, datadir):
        self.nr_of_images, self.ndims = map(int, open(os.path.join(datadir,'shape.txt')).readline().split())
        id_file = os.path.join(datadir, "id.txt")
        # python 3
        self.names = open(id_file, 'rb').read().strip().split()
        for i in range(len(self.names)):
            self.names[i] = str(self.names[i], encoding='ISO-8859-1')
        
        # python 2
        # self.names = open(id_file).read().strip().split()

        assert(len(self.names) == self.nr_of_images)
        self.name2index = dict(zip(self.names, range(self.nr_of_images)))
        self.binary_file = os.path.join(datadir, "feature.bin")
        # print ("[%s] %dx%d instances loaded from %s" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir))


    def read(self, requested, isname=True):
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert(min(requested)>=0)
            assert(max(requested)<len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []
       
        index_name_array.sort(key=lambda v:v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims
        
        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        previous = index_name_array[0][0]
 
        for next in sorted_index[1:]:
            move = (next-1-previous) * offset
            #print next, move
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = next

        fr.close()

        return [x[1] for x in index_name_array], [ res[i*self.ndims:(i+1)*self.ndims].tolist() for i in range(nr_of_images) ]


    def read_one(self, name):
        renamed, vectors = self.read([name])
        return vectors[0]    

    def shape(self):
        return [self.nr_of_images, self.ndims]


def resampling(source_length, target_length):
    return [round(i * (source_length-1) / (target_length-1)) for i in range(target_length)]


def get_frame_ids(n_total_frames, n_frames, random_type):
    if random_type == 'all_random':
        idx = random.sample([i for i in range(n_total_frames)], n_frames)
    else:
        bound = [int(i) for i in np.linspace(0, n_total_frames, n_frames+1)]
        idx = []
        for i in range(n_frames):
            if random_type == 'equally_sampling':
                tmp = (bound[i] + bound[i+1]) // 2
            else:
                tmp = np.random.randint(bound[i], bound[i+1])
            idx.append(tmp)

    return sorted(idx)

class VideoDataset(Dataset):
    def __init__(self, opt, mode, print_info=False, specific=-1, **kwargs):
        super(VideoDataset, self).__init__()
        assert mode in ['train', 'validate', 'test']
        self.opt = opt
        self.mode = mode
        if self.mode != 'train':
            self.random_type = 'equally_sampling'
            self.n_caps_per_video = 1 if not self.opt.get('parallel_mlm', False) else 0
        else:
            self.random_type = opt.get('random_type', 'segment_random')
            self.n_caps_per_video = opt.get('n_caps_per_video', 0)
            assert self.random_type in ['segment_random', 'all_random', 'equally_sampling']
            assert self.n_caps_per_video >= 0

        data = pickle.load(open(opt['info_corpus'], 'rb'))
        self.captions = data['captions']
        self.pos_tags = data['pos_tags']

        info = data['info']    
        self.itow = info['itow']
        self.itoc = info.get('itoc', None)
        self.itop = info.get('itop', None)
        self.length_info = info['length_info']
        self.splits = info['split']
        self.split_category = info.get('split_category', None)

        self.specific = specific
        self.random = np.random.RandomState(opt['seed'])

        self.databases = self._make_databases()
        self.infoset = self._make_infoset()

        if self.opt['dataset'] == 'Youtube2Text':
            self.visual_feat = BigFile(
                './data/Youtube2Text/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os'
                )
            self.video2frames = open(
                './data/Youtube2Text/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os/video2frames.txt','r'
                )
            
            a = self.video2frames.read()  
            self.dict_data = eval(a)  
            self.video2frames.close()
            
        elif self.opt['dataset']=="MSRVTT":
            # MXNet Features
            self.visual_feat = BigFile(
                './data/MSRVTT/pyresnet-152_imagenet11k,flatten0_output,os_pyresnext-101_rbps13k,flatten0_output,os'
                )
            self.video2frames = open(
                './data/MSRVTT/pyresnet-152_imagenet11k,flatten0_output,os_pyresnext-101_rbps13k,flatten0_output,os/video2frames.txt','r'
                )
            a = self.video2frames.read()
            self.dict_data = eval(a)  
            self.video2frames.close()
        
        if print_info:
            self.print_info()
    
    def print_info(self):
        print('Dataset Information:')
        print('- size of the training   set:', len(self.splits['train']))
        print('- size of the validation set:', len(self.splits['validate']))
        print('- size of the testing    set:', len(self.splits['test']))
        print('- vocab size is', len(self.itow))
        print('- maximum sequence length (\'max_len\') is set to', self.opt['max_len'])
        
        print('Modality Information:')
        for char in self.opt['modality'].lower():
            print('- loading feats_{} ({}) from {}'.format(
                char, self.opt['dim_' + char], self.opt['feats_' + char]))
        print('- load feats type: %d' % self.opt['load_feats_type'])

    def get_references(self):
        if getattr(self, 'references', None) is None:
            self.references = pickle.load(open(self.opt['reference'], 'rb'))
        return self.references

    def get_preprocessed_references(self):
        return self.captions

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.itow

    def shuffle(self):
        if self.n_caps_per_video == 0:
            # random.shuffle(self.infoset)
            pass
        else:
            self.infoset = self._make_infoset()

    def get_gt_sentences(self, vid):
        if getattr(self, 'references', None) is None:
            self.references = pickle.load(open(self.opt['reference'], 'rb'))
        return [item['caption'] for item in self.references[vid]]
    
    def get_specific_data_with_vid_and_cap_id(self, vid, cap_id, device='cpu'):
        data = self._prepare_video_features(vid)

        label = self.captions[vid][cap_id]
        tagging = self.pos_tags[vid][cap_id]
        data.update(self._prepare_input_ids(cap_id, label, tagging))

        category = self.itoc[int(vid[5:])] if self.itoc is not None else 0
        data['category'] = torch.LongTensor([category])

        for k in data.keys():
            if k not in ['frame_ids', 'video_ids', 'caption_ids']:
                data[k] = data[k].unsqueeze(0)
                data[k] = data[k].to(device)

        return data

    def _make_databases(self):
        def _load_database(path):
            if not path: return []
            if not isinstance(path, list): path = [path]
            return [h5py.File(p, 'r') for p in path if '.hdf5' in p]

        databases = []
        for char in self.opt['modality'].lower():
            key_name = "feats_%s" % char
            database = _load_database(self.opt[key_name])
            assert len(database) > 0
            databases.append([key_name, database, self.opt["dim_%s" % char]])
        return databases

    def _make_infoset(self):
        print('Preparing %s set of %s' % (self.mode, self.opt['dataset']))
        infoset = []

        # decide the size of infoset
        if self.specific != -1:
            # we only evaluate partial examples with a specific category (MSRVTT, [0, 19])
            ix_set = [int(item) for item in self.split_category[self.mode][self.specific]]
        else:
            # we evaluate all examples regardless of categories
            ix_set = [int(item) for item in self.splits[self.mode]]

        for ix in ix_set:
            vid = 'video%d' % ix
            category = self.itoc[ix] if self.itoc is not None else 0
            captions = self.captions[vid]
            pos_tags = self.pos_tags[vid] if self.pos_tags is not None else ([None] * len(captions))
            assert len(captions) == len(pos_tags)

            # prepare length info for each video example, only if decoding_type == 'NARFormmer'
            # e.g., 'video1': [0, 0, 3, 5, 0]
            if self.length_info is None:
                length_target = np.zeros(self.opt['max_len'])
            else:
                length_target = self.length_info[vid]
                length_target = length_target[:self.opt['max_len']]
                if len(length_target) < self.opt['max_len']:
                    length_target += [0] * (self.opt['max_len'] - len(length_target))

                length_target = np.array(length_target) / sum(length_target)
            
            # decide which captions are used to calculate training/evaluation loss
            if self.n_caps_per_video == 0:
                cap_id_set = [i for i in range(len(captions))]
            elif self.n_caps_per_video == 1 and self.mode != 'train':
                cap_id_set = [0]
            else:
                n_caps_per_video = min(len(captions), self.n_caps_per_video)
                cap_id_set = self.random.choice(
                    [i for i in range(len(captions))], 
                    n_caps_per_video,
                    replace=False
                )
            
            for cap_id in cap_id_set:
                item = {
                    'vid': vid,
                    'labels': captions[cap_id],
                    'pos_tags': pos_tags[cap_id],
                    'category': category,
                    'length_target': length_target,
                    'cap_id': cap_id,
                    }
                infoset.append(item)

        return infoset

    def __getitem__(self, ix):
        data = {}
        vid = self.infoset[ix]['vid']
        cap_id = self.infoset[ix]['cap_id']
        labels = self.infoset[ix]['labels']
        taggings = self.infoset[ix]['pos_tags']

        data.update(self._prepare_video_features(vid))
        data.update(self._prepare_input_ids(cap_id, labels, taggings))
        
        # some auxiliary information
        data['length_target'] = torch.FloatTensor(self.infoset[ix]['length_target'])        
        data['category'] = torch.LongTensor([self.infoset[ix]['category']])

        return data

    def __len__(self):
        return len(self.infoset)

    def _prepare_video_features(self, vid):
        _dict = {'video_ids': vid}
        
        frame_ids = get_frame_ids(
            self.opt.get('n_total_frames', 60),
            self.opt['n_frames'], 
            self.random_type
        ) if self.opt['load_feats_type'] == 0 else None

        if frame_ids is not None:
            _dict['frame_ids'] = frame_ids
            
        feats = self._make_visual_feats(self.visual_feat, self.dict_data, vid, self.opt['n_frames'])
        for info in self.databases:
            key_name = info[0]  # feats_m
            # feats = self._load_feats(info[1:], vid, frame_ids=frame_ids)
            _dict[key_name] = torch.FloatTensor(feats[:, :2048]) if key_name == 'feats_m' else torch.FloatTensor(feats[:, 2048:])
        return _dict

    def _make_visual_feats(self, visual_feat, dict_data, video_id, max_frames):
        # MXNet Features
        if self.opt['dataset'] == 'Youtube2Text': video_id = 'video' + str(int(video_id[5:]) + 1)
        frame_list = dict_data[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)
        
        video_feat = torch.zeros(max_frames, frames_tensor.size(-1), dtype=torch.float32)
        if frames_tensor.size(0) < max_frames:
            video_feat[:frames_tensor.size(0), :] = frames_tensor[:, :4096]
            
        elif frames_tensor.size(0) > max_frames:
            sample_id = np.linspace(0, frames_tensor.size(0), num=max_frames, endpoint=False, retstep=False)
            video_feat = frames_tensor[sample_id][:, :4096]
        return video_feat

    def _prepare_input_ids(self, cap_id, labels, taggings):
        _dict = {'caption_ids': cap_id}

        results = self._make_source_target(labels, taggings)
        tokens, labels, taggings = map(
            lambda x: results[x], 
            ["dec_source", "dec_target", "tagging"]
        )
        tokens_1 = results.get('dec_source_1', None)
        labels_1 = results.get('dec_target_1', None)

        _dict['tokens'] = torch.LongTensor(tokens)
        _dict['labels'] = torch.LongTensor(labels)

        if taggings is not None:
            _dict['taggings'] = torch.LongTensor(taggings)
        if tokens_1 is not None:
            _dict['tokens_1'] = torch.LongTensor(tokens_1)
            _dict['labels_1'] = torch.LongTensor(labels_1)

        return _dict

    def _load_feats(self, data, vid, **kwargs):
        frame_ids = kwargs.get('frame_ids', None)
        padding = kwargs.get('padding', True)

        databases, dim = data
        max_seq_len = databases[0].get('max_len', self.opt['n_frames'])
        if max_seq_len != self.opt['n_frames']:
            max_seq_len = int(np.asarray(max_seq_len))

        feats = []
        pre_len = None
        for database in databases:
            if vid not in database.keys():
                if padding:
                    return np.zeros((max_seq_len, dim))
                else:
                    return np.zeros(dim)
            else:
                data = np.asarray(database[vid])
                if len(data.shape) == 1 and padding:
                    if pre_len is not None:
                        data = data[np.newaxis, :].repeat(pre_len, axis=0)
                    else:
                        data = data[np.newaxis, :].repeat(self.opt.get('n_total_frames', 60), axis=0)
                else:
                    pre_len = data.shape[0]
            feats.append(data)

        if len(feats[0].shape) == 1:
            feats = np.concatenate(feats, axis=0)
            return feats

        feats = np.concatenate(feats, axis=1)

        if self.opt['load_feats_type'] == 0:
            assert frame_ids is not None
        elif self.opt['load_feats_type'] == 1:
            source_length = feats.shape[0]
            if source_length >= self.opt['n_frames']:
                frame_ids = get_frame_ids(
                        source_length, 
                        self.opt['n_frames'], 
                        self.random_type)
            else:
                frame_ids = resampling(source_length, max_seq_len)
        else:
            source_length = feats.shape[0]
            if source_length < max_seq_len:
                frame_ids = resampling(source_length, max_seq_len)
            else:
                frame_ids = [_ for _ in range(feats.shape[0])]

        return feats[frame_ids]

    def _padding(self, seq, add_eos=True):
        if seq is None:
            return None
        res = seq.copy()
        if len(res) > self.opt['max_len']:
            res = res[:self.opt['max_len']]
            if add_eos:
                res[-1] = Constants.EOS
        else:
            res += [Constants.PAD] * (self.opt['max_len'] - len(res))
        return res

    def _make_source_target(self, target, tagging):
        if self.opt['decoding_type'] == 'NARFormer':
            results = self._source_target_mlm(target[1:-1]) # exclude <bos> <eos>
        else:
            # ARFormer
            results = {
                'dec_source': self._padding(target, add_eos=True), 
                'dec_target': self._padding(target, add_eos=True)
            }

        assert len(results['dec_source']) == len(results['dec_target'])

        if self.opt.get('visual_word_generation', False):
            results.update(self._source_target_visual_word(target=target, pos_tag=tagging))

        if 'tagging' not in results.keys():
            results['tagging'] = self._padding(tagging, add_eos=True)

        return results

    def _source_target_mlm(self, target):
        assert target[0] != Constants.BOS
        assert target[-1] != Constants.EOS

        beta_low, beta_high = self.opt.get('beta', [0, 1])

        min_num_masks = 1
        dec_source = torch.LongTensor(target)
        dec_target_cp = torch.LongTensor(target)
        dec_target = torch.LongTensor([Constants.PAD] * len(dec_source))

        if self.mode == 'train':
            if min_num_masks >= len(dec_source):
                ind = np.array([],dtype=np.uint8)
            else:
                low = max(int(len(dec_source) * beta_low), min_num_masks)
                high = max(int(len(dec_source) * beta_high), min_num_masks)
                if high == low:
                    high += 1
                sample_size = self.random.randint(low, high)
                ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
            
            if len(ind):
                dec_source[ind] = Constants.MASK
                dec_target[ind] = dec_target_cp[ind]
        else:
            dec_source[dec_source!=Constants.PAD] = Constants.MASK
            dec_target = dec_target_cp           

        dec_source = self._padding(dec_source.tolist(), add_eos=False)
        dec_target = self._padding(dec_target.tolist(), add_eos=False)
        
        return {'dec_source': dec_source, 'dec_target': dec_target}

    def _source_target_visual_word(self, **kwargs):
        target = kwargs['target']
        pos_tag = kwargs['pos_tag']
        sent_length = len(target[1:-1]) # exclude <bos> <eos>

        visual_tag = Constants.VIS
        target_tag = Constants.MASK

        if self.mode != 'train':
            dec_target_1 = [0]
            dec_source_1 = [0]
        else:
            assert len(target) == len(pos_tag)
            assert self.itop is not None

            dec_source_1 = self._padding(
                [visual_tag] * (sent_length if self.opt['decoding_type'] == 'NARFormer' else len(target)), 
                add_eos=False if self.opt['decoding_type'] == 'NARFormer' else True
            )

            # get the position of tokens that have the pos_tag we demand
            pos_satisfied_ind = []
            for i, item in enumerate(pos_tag[1:-1]):
                w = self.itow[target[i+1]]
                # we ignore verb ``be''
                if self.itop[item] in self.opt['demand'] and w not in ['is', 'are', 'was', 'were', 'be']:
                    pos_satisfied_ind.append(i)

            pos_satisfied_ind = np.array(pos_satisfied_ind)
            
            # decoder1 need to predict tokens with satisfied pos_tag from scratch
            # meanwhile, decoder1 should learn to keep the remaining tokens (i.e., <mask>) unchanged
            dec_target_1 = torch.LongTensor([target_tag] * sent_length)
            dec_target_cp = torch.LongTensor(target[1:-1])
            dec_target_1[pos_satisfied_ind] = dec_target_cp[pos_satisfied_ind]

            if self.opt['decoding_type'] == 'NARFormer':
                dec_target_1 = self._padding(dec_target_1.tolist(), add_eos=False)
            else:
                # when training with autoregressive transformer, the first token will be ignored, i.e., label = dec_target_1[1:]
                dec_target_1 = self._padding([target[0]] + dec_target_1.tolist() + [Constants.EOS], add_eos=True)

        return {'dec_source_1': dec_source_1, 'dec_target_1': dec_target_1}

