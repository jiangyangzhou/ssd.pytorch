"""CrowdHuman Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot, Yangzhou Jiang
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

CH_CLASSES = (  # always index 0
    'person','head')

# note: if you used our download scripts, this should be right
CH_ROOT = osp.join(HOME, "Passport/jyz/data/crowdHuman/")


class CrowdHumanAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CH_CLASSES, range(len(CH_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target,height,width):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class CrowdHumanDetection(data.Dataset):
    """CrowdHuman Detection Dataset Object
        (cause I have converted crowdhuman dataset to the voc form, so it's based on voc script)
    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, annopath,
                 transform=None, 
                 dataset_name='crowdHuman'):
        self.root = root
        self.transform = transform
        self.name = dataset_name
        self.anno=[]
        with open(annopath,'r') as f:
            while True:
                d = f.readline()
                if not d:
                    break
                self.anno.append(json.loads(d))
        print("root is:",root)
        self._imgpath = osp.join(root+'/Images', '%s.jpg')
        print("self._imgpath is:",self._imgpath)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.anno)

    def pull_item(self, index):
        img_info = self.anno[index]
        img_id = self.anno[index]['ID']
        img = cv2.imread(self._imgpath % img_id)
        print(img.shape)
        height,width,channel = img.shape
        img_boxes = self.anno[index]['gtboxes']
        box_list = list()
        for b in img_boxes:
            if b['tag']=='person':
                hbox=b['hbox'][:2]+ [b['hbox'][0]+b['hbox'][2], b['hbox'][1]+b['hbox'][3]]
                if 'head_attr' in b.keys() and 'ignore' in b['head_attr'].keys() and b['head_attr']['ignore']==1:
                    hbox = [0,0,0,0]
                    print("ignore head")
                fbox= b['fbox'][:2]+ [b['fbox'][0]+b['fbox'][2], b['fbox'][1]+b['fbox'][3]]
                if 'extra' in b.keys() and 'extra' in b['extra'].keys() and b['extra']['ignore']==1:
                    fbox = [0,0,0,0]
                    print("ignore body")
                box_list.append(fbox+hbox+[1])
        if self.transform is not None:
            target = np.array(box_list,dtype='float')
            box_num = len(box_list)
            print("target's shape is",target.shape, "box_num=",box_num)
            img, boxes, labels = self.transform(img, np.vstack((target[:, :4], target[:,4:8])), np.ones((2*box_num,1)) )
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            print("box and label shape1:",boxes.shape,labels.shape)
            try:
                boxes = np.hstack((boxes[:box_num,:],boxes[box_num:,:]))
            except ValueError as e:
                print(box_num,boxes.shape,e)
            labels = labels[:box_num]
            print("box and label shape2:",boxes.shape,labels.shape)
            target = np.hstack((boxes, labels))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_name = self.anno[index]['ID']
        img = cv2.imread(self._imgpath % img_name)
        return img
        
    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        anno = self.anno[index]
        box_list = list()
        img_boxes=anno['gt_boxes']
        for b in img_boxes:
            if b['tag']=='person':
                hbox=b['hbox'][:2]+ [b['hbox'][0]+b['hbox'][2], b['hbox'][1]+b['hbox'][3]]
                if 'head_attr' in b.keys() and b['head_attr']['ignore']==1:
                    hbox = [-1,0,0,0]
                    print("ignore head")
                fbox= b['fbox'][:2]+ [b['fbox'][0]+b['fbox'][2], b['fbox'][1]+b['fbox'][3]]
                if 'extra' in b.keys() and b['head_attr']['ignore']==1:
                    fbox = [-1,0,0,0]
                    print("ignore body")
                box_list.append(['person',tuple(fbox+hbox)])
        return (anno['ID'],box_list)

 
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)



