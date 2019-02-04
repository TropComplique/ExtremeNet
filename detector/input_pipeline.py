import cv2
import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset


class ExtremePointsDataset(Dataset):

    def __init__(self, coco, image_folder, is_training=False, training_size=None):
        """
        During the training I use square images.
        But during the evaluation I only pad images with zeros.
        And I want all my image sizes be divisible by 32.

        Arguments:
            coco: an instance of COCO.
            image_folder: a string.
            is_training: a boolean.
            training_size: an integer or None.
        """
        self.labels = coco.getCatIds(catNms=['person'])
        self.ids = coco.getImgIds(catIds=self.labels)
        self.coco = coco
        self.image_folder = image_folder
        self.is_training = is_training

        if is_training:
            assert training_size % 32 == 0
            self.training_size = training_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Returns:
            image: a float tensor with shape [3, h, w].
                It represents a RGB image with
                pixel values in the range [0, 1].
            heatmaps: a float tensor with shape [5, h/4, w/4].
            offsets: a float tensor with shape [10, h/4, w/4].
            masks: a float tensor with shape [2, h/4, w/4].
            num_boxes: a long tensor with shape [].
        """

        # LOAD AN IMAGE AND ANNOTATIONS

        image_id = self.ids[index]
        filename = self.coco.loadImgs(image_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.image_folder, filename))

        # sometimes images are gray
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # load all person annotations
        annotations_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.labels, iscrowd=None)
        annotations = self.coco.loadAnns(annotations_ids)

        # OPTIONALLY DO A RANDOM CROP

        if self.is_training:
            # choose a random window:
            min_dimension = min(height, width)
            size = np.random.randint(min_dimension // 2, min_dimension)
            x = np.random.randint(0, width - size)
            y = np.random.randint(0, height - size)
            height, width = size, size  # new size

            ymin, xmin, ymax, xmax = y, x, y + size, x + size
            image = image[ymin:ymax, xmin:xmax]
        else:
            # whole image window:
            ymin, xmin, ymax, xmax = 0, 0, height, width

        # DETECT EXTREME POINTS AND COLLECT ANNOTATIONS INTO ARRAYS

        segmentation_mask = np.zeros([height, width], dtype='bool')
        not_ignore_mask = np.ones([height, width], dtype='bool')  # mask for loss

        extreme_points = []
        for a in annotations:

            mask = self.coco.annToMask(a)
            mask = mask[ymin:ymax, xmin:xmax]

            unannotated = a['iscrowd'] == 1
            too_small = mask.sum() < 100  # whether area is too small

            if too_small or unannotated:
                not_ignore_mask = np.logical_and(mask == 0, not_ignore_mask)
                continue

            extreme_points.append(get_extreme_points(mask))
            segmentation_mask = np.logical_or(segmentation_mask, mask == 1)

        # number of persons on the image
        num_boxes = len(extreme_points)

        if num_boxes == 0:
            extreme_points = np.empty([0, 4, 2], dtype='int32')
        else:
            extreme_points = np.stack(extreme_points, axis=0)
            # it has shape [num_boxes, 4, 2]

        masks = np.stack([segmentation_mask, not_ignore_mask], axis=2)  # shape [height, width, 2]
        image_and_masks = np.concatenate([image, masks], axis=2)  # shape [height, width, 5]

        # RESIZE THE IMAGE AND ANNOTATIONS (AND OPTIONALLY AUGMENT DATA)

        if self.is_training:
            image_and_masks, extreme_points = random_flip(image_and_masks, extreme_points)
            w, h = self.training_size, self.training_size
            image_and_masks = cv2.resize(image_and_masks, (w, h), cv2.INTER_NEAREST)
            scaler = np.array([w/width, h/height])
            extreme_points = (extreme_points * scaler).astype('int32')
        else:
            # pad the image and masks with zeros
            w, h = 32 * math.ceil(width/32), 32 * math.ceil(height/32)
            empty = np.zeros((h, w, 5), dtype='uint8')
            empty[:height, :width] = image_and_masks
            image_and_masks = empty

        # DOWNSAMPLE ANNOTATIONS AND CREATE HEATMAPS FOR REGRESSION

        # find centers
        t, b, l, r = np.split(extreme_points, 4, axis=1)
        cx, cy = (l[:, :, 0] + r[:, :, 0]) // 2, (t[:, :, 1] + b[:, :, 1]) // 2
        # they have shape [num_boxes, 1]

        center = np.stack([cx, cy], axis=2)  # shape [num_boxes, 1, 2]
        extreme_points = np.concatenate([extreme_points, center], axis=1)

        image, masks = np.split(image_and_masks, indices_or_sections=[3], axis=2)
        w, h = w // 4, h // 4  # downsampled size
        masks = cv2.resize(masks, (w, h), cv2.INTER_NEAREST)
        offsets = (extreme_points / 4.0) - (extreme_points // 4)  # shape [num_boxes, 5, 2]
        extreme_points = extreme_points // 4  # floor

        heatmaps, vectormaps = [], []
        for i in range(5):

            # get a particular point type
            points = extreme_points[:, i]

            heatmaps.append(generate_heatmap(points, h, w))
            # don't forget to convert to (y, x) format:
            values = torch.FloatTensor(offsets[:, i, [1, 0]])
            index = torch.LongTensor(points[:, [1, 0]])
            vectormap = torch.sparse.FloatTensor(index.t(), values, [h, w, 2]).to_dense()
            vectormaps.append(vectormap)

        # CONVERT TO PYTORCH TENSORS

        image = torch.FloatTensor(image/255.0).permute(2, 0, 1)
        heatmaps = torch.FloatTensor(np.stack(heatmaps, axis=0))
        offsets = torch.cat(vectormaps, dim=2).permute(2, 0, 1)
        masks = torch.FloatTensor(masks).permute(2, 0, 1)
        num_boxes = torch.tensor(num_boxes)

        return image, heatmaps, offsets, masks, num_boxes


def get_extreme_points(mask):
    """
    Arguments:
        mask: a numpy uint8 array with shape [h, w].
    Returns:
        a numpy int array with shape [4, 2].
        It represents points [top, bottom, left, right] in format [x, y].
        Coordinates are in ranges [0, w - 1] and [0, h - 1].
    """
    threshold = 0.02

    points = np.where(mask > 0)
    points = np.asarray(points, dtype='int32').transpose()  # shape [num_points, 2]
    y, x = points[:, 0], points[:, 1]

    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    h, w = ymax - ymin, xmax - xmin

    # top
    near_x = x[np.abs(y - ymin) <= threshold * h]
    tx = (near_x.max() + near_x.min()) // 2
    ty = ymin

    # bottom
    near_x = x[np.abs(y - ymax) <= threshold * h]
    bx = (near_x.max() + near_x.min()) // 2
    by = ymax

    # left
    near_y = y[np.abs(x - xmin) <= threshold * w]
    lx = xmin
    ly = (near_y.max() + near_y.min()) // 2

    # right
    near_y = y[np.abs(x - xmax) <= threshold * w]
    rx = xmax
    ry = (near_y.max() + near_y.min()) // 2

    return np.array([[tx, ty], [bx, by], [lx, ly], [rx, ry]], dtype='int32')


def generate_heatmap(points, h, w):
    """
    Arguments:
        points: a numpy int array with shape [num_points, 2].
            It is in the form [x, y].
            And it is assumed that `0 <= x < w` and `0 <= y < h`.
        h, w: integers.
    Returns:
        a numpy float array with shape [h, w].
    """

    k = 11  # sigmas mustn't be too large!
    a = np.arange(-k, k + 1, dtype='float32')
    X, Y = np.meshgrid(a, a)  # they have shape [2*k + 1, 2*k + 1]
    D = 0.5 * (X**2 + Y**2)
    heatmap = np.zeros([h + 2*k, w + 2*k], dtype='float32')

    for x, y in points:
        sigma = 1.0
        g = np.exp(-D/(sigma**2))
        ymin, ymax = y, y + 2*k + 1
        xmin, xmax = x, x + 2*k + 1
        heatmap[ymin:ymax, xmin:xmax] = np.maximum(g, heatmap[ymin:ymax, xmin:xmax])

    return heatmap[k:-k, k:-k]


def random_flip(image_and_masks, extreme_points):

    if np.random.rand() > 0.5:

        image_and_masks = cv2.flip(image_and_masks, 1)
        height, width, _ = image_and_masks.shape
        x, y = np.split(extreme_points, 2, axis=2)
        extreme_points = np.concatenate([width - x - 1, y], axis=2)

        # switch left and right
        correct_order = [0, 1, 3, 2]
        extreme_points = extreme_points[:, correct_order]

    return image_and_masks, extreme_points
