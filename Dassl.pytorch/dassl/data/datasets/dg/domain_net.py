import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

import random

@DATASET_REGISTRY.register()
class DomainNet(DatasetBase):
    """DomainNet.

    Statistics:
        - six domains: clipart, infograph, painting, quickdraw, real, sketch
        - This dataset contains 586; 575 examples of size (3; 224; 224) and 345 classes.

    Reference:
        - Xingchao et al. Moment Matching for Multi-Source Domain Adaptation.
    """

    dataset_dir = "domainnet"
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, "test")
        length_test = len(test)
    
        # val = random.sample(test, length_test//10)
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="test")
        
        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                file_train = osp.join(
                    self.split_dir, dname + "_train.txt"
                )
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(
                    self.split_dir, dname + "_test.txt"
                )
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(
                    self.split_dir, dname + "_" + split + ".txt"
                )
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                classname = impath.split("/")[-2]
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.image_dir, impath)
                label = int(label)
                items.append((impath, label))

        return items