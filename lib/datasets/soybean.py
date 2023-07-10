import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy import spatial

from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, fast_hist, per_class_iu

CLASS_LABELS = ('stem', 'leaf', 'soil')
VALID_CLASS_IDS = (0, 1)
SCANNET_COLOR_MAP = {
    0: (23., 190., 207.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
}



class soybeanVoxelizationDataset(VoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.05

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 3  # Will be converted to 20 as defined in IGNORE_LABELS.
  IGNORE_LABELS = tuple(set(range(3)) - set(VALID_CLASS_IDS))
  IS_FULL_POINTCLOUD_EVAL = True

  # If trainval.txt does not exist, copy train.txt and add contents from val.txt
  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train.txt',
      DatasetPhase.Val: 'val.txt',
      DatasetPhase.TrainVal: 'trainval.txt',
      DatasetPhase.Test: 'test.txt'
  }

  def __init__(self,
               config,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    data_root = config.soybean_path
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_paths = read_txt(os.path.join('./splits/soybean', self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

  def _augment_locfeat(self, pointcloud):
    # Assuming that pointcloud is xyzrgb(...), append location feat.
    pointcloud = np.hstack(
        (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
         pointcloud[:, 6:]))
    return pointcloud

  def test_pointcloud(self, pred_dir):
    print('Running full pointcloud evaluation.')
    eval_path = os.path.join(pred_dir, 'fulleval')
    os.makedirs(eval_path, exist_ok=True)
    # Join room by their area and room id.
    # Test independently for each room.
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
    for i, data_path in enumerate(self.data_paths):
      room_id = self.get_output_id(i)
      pred = np.load(os.path.join(pred_dir, 'pred_%04d_%02d.npy' % (i, 0)))

      # save voxelized pointcloud predictions
      save_point_cloud(
          np.hstack((pred[:, :3], np.array([SCANNET_COLOR_MAP[i] for i in pred[:, -1]]))),
          f'{eval_path}/{room_id}_voxel.ply',
          verbose=False)

      fullply_f = self.data_root / data_path
      query_pointcloud = read_plyfile(fullply_f)
      query_xyz = query_pointcloud[:, :3]
      query_label = query_pointcloud[:, -1]
      # Run test for each room.
      pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
      _, result = pred_tree.query(query_xyz)
      ptc_pred = pred[result, 3].astype(int)
      # Save prediciton in txt format for submission.
      np.savetxt(f'{eval_path}/{room_id}.txt', ptc_pred, fmt='%i')
      # Save prediciton in colored pointcloud for visualization.
      save_point_cloud(
          np.hstack((query_xyz, np.array([SCANNET_COLOR_MAP[i] for i in ptc_pred]))),
          f'{eval_path}/{room_id}.ply',
          verbose=False)
      # Evaluate IoU.
      if self.IGNORE_LABELS is not None:
        ptc_pred = np.array([self.label_map[x] for x in ptc_pred], dtype=np.int)
        query_label = np.array([self.label_map[x] for x in query_label], dtype=np.int)
      hist += fast_hist(ptc_pred, query_label, self.NUM_LABELS)
    ious = per_class_iu(hist) * 100
    print('mIoU: ' + str(np.nanmean(ious)) + '\n'
          'Class names: ' + ', '.join(CLASS_LABELS) + '\n'
          'IoU: ' + ', '.join(np.round(ious, 2).astype(str)))


class soybeanVoxelization5cmDataset(soybeanVoxelizationDataset):
  VOXEL_SIZE = 0.002