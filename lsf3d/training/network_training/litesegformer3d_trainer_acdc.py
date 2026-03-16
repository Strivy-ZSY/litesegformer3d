#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from lsf3d.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from lsf3d.training.loss_functions.deep_supervision import MultipleOutputLoss2
from lsf3d.utilities.to_torch import maybe_to_torch, to_cuda
from lsf3d.network_architecture.initialization import InitWeights_He
from lsf3d.network_architecture.neural_network import SegmentationNetwork
from lsf3d.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from lsf3d.training.dataloading.dataset_loading import unpack_dataset
from lsf3d.training.network_training.Trainer_acdc import Trainer_acdc
from lsf3d.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from lsf3d.training.learning_rate.poly_lr import poly_lr
from lsf3d.training.trainable_params_format.py import format_params
from batchgenerators.utilities.file_and_folder_operations import *
from lsf3d.network_architecture.litesegformer3d_acdc import LiteSegFormer3D
from fvcore.nn import FlopCountAnalysis


class litesegformer3d_trainer_acdc(Trainer_acdc):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        # self.max_num_epochs = 1
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None # 深度监督尺度
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight = False

        # 添加这些属性
        self.deep_supervision = True
        self.do_ds = True
        self.network = None

        self.load_plans_file()

        if len(self.plans['plans_per_stage']) == 2:
            Stage = 1
        else:
            Stage = 0

        self.crop_size = self.plans['plans_per_stage'][Stage]['patch_size']
        self.input_channels = self.plans['num_modalities']
        self.num_classes = int(self.plans['num_classes'] + 1)  # 修复：从plans文件动态获取
        self.conv_op = nn.Conv3d

        self.embedding_dim = 96
        self.depths = [2, 2, 2, 2]
        self.num_heads = [3, 6, 12, 24]
        self.embedding_patch_size = [1, 4, 4]
        self.window_size = [[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]]
        self.down_stride = [[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]]
        self.deep_supervision = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.plans['plans_per_stage'][0]['patch_size'] = np.array([16, 160, 160])
            self.crop_size = np.array([16, 160, 160])

            self.plans['plans_per_stage'][self.stage]['pool_op_kernel_sizes'] = [[1, 4, 4], [2, 2, 2], [2, 2, 2]]
            self.process_plans(self.plans)

            self.setup_DA_params()
            if self.deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                self.ds_loss_weights = weights
                # now wrap the loss
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory,
                                                      self.plans['data_identifier'] + "_stage%d" % self.stage)
            seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """Initialize the LiteSegFormer3D network"""
        # 初始化网络
        self.network = LiteSegFormer3D(
            in_channels=self.input_channels,
            num_classes=self.num_classes,
            embed_dims=[32, 64, 128, 256],
            patch_kernel_size=[3, 3, 3, 3],
            patch_stride=[2, 2, 2, 2],
            patch_padding=[1, 1, 1, 1],
            depths=[2, 2, 2, 2],
            num_heads=[1, 2, 4, 8],
            sr_ratios=[8, 4, 2, 1],
        )

        # 确保do_ds属性被正确设置
        self.network.do_ds = True
        # 确保num_classes属性同步
        self.network.num_classes = self.num_classes

        if torch.cuda.is_available():
            self.network.cuda()
        input_res = (self.input_channels, 16, 160, 160)
        self.network.inference_apply_nonlin = softmax_helper

        from torchsummary import summary
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        summary(self.network, input_size = input_res, device='cuda' if torch.cuda.is_available() else 'cpu')

        sys.stdout = old_stdout
        output = mystdout.getvalue()
        lines = output.strip().split('\n')

        line = lines[-3:-2][0]
        # print(line)
        print(format_params(line))
        

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        Ensure target is in the correct format for evaluation.
        """
        # Handle deep supervision case
        if isinstance(target, list):
            target = target[0]  # Use the highest resolution target
        target = target.long()  # Ensure target is a tensor and has the correct type
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """

        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu,segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        # Handle target for deep supervision
        if isinstance(target, list):
            target_list = [t.long() for t in target]
        else:
            target = target.long()
            target_list = [target]

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data

                if isinstance(output, list):
                    resized_target_list = [
                        torch.nn.functional.interpolate(
                            target_list[0].float(), size=o.shape[2:], mode='nearest'
                        ).long() for o in output
                    ]
                    output_for_eval = output[0]
                else:
                    output_for_eval = output
                    resized_target_list = target_list

                l = self.loss(output, resized_target_list)
        else:
            output = self.network(data)
            del data

            if isinstance(output, list):
                resized_target_list = [
                    torch.nn.functional.interpolate(
                        target_list[0].float(), size=o.shape[2:], mode='nearest'
                    ).long() for o in output
                ]
                output_for_eval = output[0]
            else:
                output_for_eval = output
                resized_target_list = target_list

            l = self.loss(output, resized_target_list)

        if do_backprop:
            if self.fp16:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
            else:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output_for_eval, target_list[0])

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        默认的划分是对所有可用的训练案例进行5折交叉验证。nnU-Net将创建一个划分（它是有种子的，因此始终相同），
        并将其保存为splits_final.pkl文件在预处理数据目录中。
        有时您可能希望出于各种原因创建自己的划分。为此，您需要创建自己的splits_final.pkl文件。
        如果此文件存在，nn-U-Net将使用它并使用其中定义的任何划分。您可以在此文件中创建任意数量的划分。
        请注意，如果您仅定义了4个划分（fold 0-3），然后在训练时设置fold=4（即第五个划分），
        nn-U-Net将打印警告并继续使用随机的80:20数据划分。
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            splits[self.fold]['train']=np.array(['patient001_frame01', 'patient001_frame12', 'patient004_frame01',
       'patient004_frame15', 'patient005_frame01', 'patient005_frame13',
       'patient006_frame01', 'patient006_frame16', 'patient007_frame01',
       'patient007_frame07', 'patient010_frame01', 'patient010_frame13',
       'patient011_frame01', 'patient011_frame08', 'patient013_frame01',
       'patient013_frame14', 'patient015_frame01', 'patient015_frame10',
       'patient016_frame01', 'patient016_frame12', 'patient018_frame01',
       'patient018_frame10', 'patient019_frame01', 'patient019_frame11',
       'patient020_frame01', 'patient020_frame11', 'patient021_frame01',
       'patient021_frame13', 'patient022_frame01', 'patient022_frame11',
       'patient023_frame01', 'patient023_frame09', 'patient025_frame01',
       'patient025_frame09', 'patient026_frame01', 'patient026_frame12',
       'patient027_frame01', 'patient027_frame11', 'patient028_frame01',
       'patient028_frame09', 'patient029_frame01', 'patient029_frame12',
       'patient030_frame01', 'patient030_frame12', 'patient031_frame01',
       'patient031_frame10', 'patient032_frame01', 'patient032_frame12',
       'patient033_frame01', 'patient033_frame14', 'patient034_frame01',
       'patient034_frame16', 'patient035_frame01', 'patient035_frame11',
       'patient036_frame01', 'patient036_frame12', 'patient037_frame01',
       'patient037_frame12', 'patient038_frame01', 'patient038_frame11',
       'patient039_frame01', 'patient039_frame10', 'patient040_frame01',
       'patient040_frame13', 'patient041_frame01', 'patient041_frame11',
       'patient043_frame01', 'patient043_frame07', 'patient044_frame01',
       'patient044_frame11', 'patient045_frame01', 'patient045_frame13',
       'patient046_frame01', 'patient046_frame10', 'patient047_frame01',
       'patient047_frame09', 'patient050_frame01', 'patient050_frame12',
       'patient051_frame01', 'patient051_frame11', 'patient052_frame01',
       'patient052_frame09', 'patient054_frame01', 'patient054_frame12',
       'patient056_frame01', 'patient056_frame12', 'patient057_frame01',
       'patient057_frame09', 'patient058_frame01', 'patient058_frame14',
       'patient059_frame01', 'patient059_frame09', 'patient060_frame01',
       'patient060_frame14', 'patient061_frame01', 'patient061_frame10',
       'patient062_frame01', 'patient062_frame09', 'patient063_frame01',
       'patient063_frame16', 'patient065_frame01', 'patient065_frame14',
       'patient066_frame01', 'patient066_frame11', 'patient068_frame01',
       'patient068_frame12', 'patient069_frame01', 'patient069_frame12',
       'patient070_frame01', 'patient070_frame10', 'patient071_frame01',
       'patient071_frame09', 'patient072_frame01', 'patient072_frame11',
       'patient073_frame01', 'patient073_frame10', 'patient074_frame01',
       'patient074_frame12', 'patient075_frame01', 'patient075_frame06',
       'patient076_frame01', 'patient076_frame12', 'patient077_frame01',
       'patient077_frame09', 'patient078_frame01', 'patient078_frame09',
       'patient080_frame01', 'patient080_frame10', 'patient082_frame01',
       'patient082_frame07', 'patient083_frame01', 'patient083_frame08',
       'patient084_frame01', 'patient084_frame10', 'patient085_frame01',
       'patient085_frame09', 'patient086_frame01', 'patient086_frame08',
       'patient087_frame01', 'patient087_frame10'])
            splits[self.fold]['val']=np.array(['patient089_frame01', 'patient089_frame10', 'patient090_frame04',
       'patient090_frame11', 'patient091_frame01', 'patient091_frame09',
       'patient093_frame01', 'patient093_frame14', 'patient094_frame01',
       'patient094_frame07', 'patient096_frame01', 'patient096_frame08',
       'patient097_frame01', 'patient097_frame11', 'patient098_frame01',
       'patient098_frame09', 'patient099_frame01', 'patient099_frame09',
       'patient100_frame01', 'patient100_frame13'])
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
