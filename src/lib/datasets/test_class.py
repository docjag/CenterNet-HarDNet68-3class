from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from sample.ctdet import CTDetDataset
from dataset.coco import COCO


# Dictionary of Datasets Objects
dataset_factory = {
  'coco': COCO,
}

_sample_factory = {
  'ctdet': CTDetDataset,

}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset


class Dataset(object):
  """docstring for Dataset"""
  def __init__(self, dataset, task):
    super(Dataset, self).__init__()
    
    self.dataset = dataset_factory[dataset]
    self.task = _sample_factory[task]
    


# test = get_dataset('coco', 'ctdet')
# print(dir(test))

test = Dataset('coco', 'ctdet')
print(dir(test.dataset))
print(dir(test.task))