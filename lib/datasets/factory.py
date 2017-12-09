# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

import datasets.pascal_voc
from datasets.transfer_pascal_voc import transfer_pascal_voc
import datasets.kitti
import datasets.nthu
import datasets.transfer

import pdb

__sets = {}

# Set up voc_<year>_<split>
for year in ['2007']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year, data_path=None:
                datasets.pascal_voc(split, year, data_path))

# VOC transfer learning
for split in ['trainval']:
    for is_source in [True, False]:
        domain = "source" if is_source else "target"
        name = 'transfer_{}_{}'.format(domain, split)
        __sets[name] = (lambda split=split, is_source=is_source, data_path=None:
                transfer_pascal_voc(split, is_source, data_path))

# KITTI dataset
for split in ['train', 'val', 'trainval', 'test']:
    name = 'kitti_{}'.format(split)
    __sets[name] = (lambda split=split, data_path=None:
            datasets.kitti(split))

# NTHU dataset
for split in ['71', '370']:
    name = 'nthu_{}'.format(split)
    __sets[name] = (lambda split=split, data_path=None:
            datasets.nthu(split, data_path))

def get_imdb(name, data_path=None):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    #pdb.set_trace()
    return __sets[name](data_path=data_path)

def get_transfer_imdb(source, target,
            source_data_path=None, target_data_path=None):
    """
    Get an imdb for transfer learning.
    Returned imdb has access to both domains' data.
    """
    if not source in __sets:
        raise KeyError('Unknown dataset: {}'.format(source))
    if not target in __sets:
        raise KeyError('Unknown dataset: {}'.format(target))
    source_imdb = __sets[source](data_path=source_data_path)
    target_imdb = __sets[target](data_path=target_data_path)
    return datasets.transfer(source_imdb, target_imdb)

def list_imdbs():
    """List all registered imdbs."""
