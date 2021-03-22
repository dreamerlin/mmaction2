from mmcv.runner import HOOKS
from mmcv.runner import Hook

import os
import os.path as osp
import shutil

import mmcv


@HOOKS.register_module()
class SaveImgHook(Hook):

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def get_mode(self, runner):
        if runner.mode == 'train':
            if 'time' in runner.log_buffer.output:
                mode = 'train'
            else:
                mode = 'val'
        elif runner.mode == 'val':
            mode = 'val'
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return mode

    def get_epoch(self, runner):
        if runner.mode == 'train':
            epoch = runner.epoch + 1
        elif runner.mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch

    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def after_train_iter(self, runner):
        epoch = self.get_epoch(runner)
        selected_frame_names = runner.outputs.get('selected_frame_names', None)
        if selected_frame_names is None:
            return

        for items in selected_frame_names:
            if items is None:
                continue
            for i, img_path in enumerate(items):
                img_path_0 = osp.basename(img_path)
                img_path_0 = f'{i:02}' + '_' + img_path_0
                img_dir_1 = osp.basename(osp.dirname(img_path))
                img_path_1_0 = osp.join(self.save_dir, str(epoch) + '_epoch', img_dir_1, img_path_0)
                mmcv.mkdir_or_exist(osp.dirname(img_path_1_0))
                shutil.copy(img_path, img_path_1_0)
            print('Saving Done')