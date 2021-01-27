import copy
import os.path as osp

import numpy as np
import torch

from .rawframe_dataset import RawframeDataset
from .registry import DATASETS


@DATASETS.register_module()
class NovelDataset(RawframeDataset):

    def load_annotations(self):
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}

                frame_dir = line_split[0]
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)

                frame_inds = list(map(int, line_split[1].split(',')))
                frame_inds = np.array(frame_inds)
                label = int(line_split[2])

                video_info['frame_dir'] = frame_dir
                video_info['frame_inds'] = frame_inds
                video_info['label'] = label

                video_infos.append(video_info)

        return video_infos

    def prepare_train_frames(self, idx):
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])

        results['clip_len'] = 1
        results['frame_interval'] = 1
        results['num_clips'] = len(results['frame_inds'])

        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        results['clip_len'] = 1
        results['frame_interval'] = 1
        results['num_clips'] = len(results['frame_inds'])

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)
