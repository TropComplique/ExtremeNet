import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ExtremePointsDataset(Dataset):

    def __init__(self, is_training, coco):

        self.labels = coco.getCatIds(catNms=['person'])
        self.ids = coco.getImgIds(catIds=self.labels)
        self.coco = coco
        self.image_folder = ''
        self.training_size = (640, 640)

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

        image_id = self.ids[index]
        filename = coco.loadImgs(image_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.image_folder, filename))
        height, width, _ = image.size

        annotations_ids = coco.getAnnIds(imgIds=image_id, catIds=self.labels, iscrowd=None)
        annotations = coco.loadAnns(annIds)

        segmentation_mask = np.zeros([height, width], dtype='bool')
        not_ignore_mask = np.ones([height, width], dtype='bool')  # mask for loss

        extreme_points = []
        for a in annotations:

            if a['is_crowd'] == 1:
                unannotated_person_mask = coco.annToMask(a)
                use_this = unannotated_person_mask == 0
                not_ignore_mask = np.logical_and(use_this, not_ignore_mask)
                continue

            mask = self.coco.annToMask(a)
            extreme_points.append(get_extreme_points(mask))
            segmentation_mask = np.logical_or(segmentation_mask, mask == 1)

        masks = np.stack([segmentation_mask, not_ignore_mask], axis=2)  # shape [height, width, 2]
        extreme_points = np.stack(extreme_points, axis=0)  # shape [num_boxes, 5, 2]
        image_and_masks = np.concat([image, masks], axis=2)  # shape [height, width, 5]

        if is_training:

            image_and_masks, extreme_points = random_crop(image_and_masks, extreme_points)
            # they have shapes [height, width, 5] and [num_boxes, 5, 2],
            # note that `num_boxes`, `height`, and `width` could change here!
            # also point coordinates are in ranges [0, width - 1] and [0, height - 1]

            image_and_masks, extreme_points = random_flip(image_and_masks, extreme_points)
            image_and_masks = random_color_jitter(image_and_masks)
            w, h = self.training_size  # they are divisible by 4
        else:
            empty = np.zeros((height, width, 3), dtype=image.dtype)
            image_and_masks
            image_and_masks = resize_keeping_aspect_ratio(
                image_and_masks, min_dimension=640, max_dimension=None)
            w, h = size  # they are divisible by 4

        height, width, _ = image_and_masks.shape
        image_and_masks = cv2.resize(image_and_masks, (w, h), cv2.INTER_NEAREST)
        image, masks = np.split(image_and_masks, indices_or_sections=[3], axis=2)
        scaler = np.array([w/width, h/height])
        extreme_points = (extreme_points * scaler).astype('int32')

        masks = cv2.resize(masks, (w // 4, h // 4), cv2.INTER_NEAREST)
        offsets = (extreme_points / 4.0) - (extreme_points // 4)  # shape [num_boxes, 5, 2]
        extreme_points = extreme_points // 4  # floor
        num_boxes = torch.tensor(len(offsets))

        heatmaps, vectormaps = [], []
        for i in range(5):
            points = extreme_points[:, i]
            heatmaps.append(generate_heatmap(points, h // 4, w // 4))

            values = torch.FloatTensor(offsets[:, i])
            index = torch.LongTensor(extreme_points[:, i])
            vectormap = torch.sparse.FloatTensor(index.t(), values, [h // 4, w // 4, 2]).to_dense()
            vectormaps.append(vectormap)

        heatmaps = torch.FloatTensor(np.stack(heatmaps, axis=0))  # shape [5, h // 4, w // 4]
        offsets = torch.cat(vectormaps, dim=2).permute(2, 0, 1)  # shape [10, h // 4, w // 4]

        return image, heatmaps, offsets, masks, num_boxes


def get_extreme_points(mask):
    """
    Arguments:
        mask: a numpy uint8 array with shape [h, w].
    Returns:
        a numpy int array with shape [5, 2].
        It represents points [top, bottom, left, right, center] in format [x, y].
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

    # center
    cx = (lx + rx) // 2
    cy = (ty + by) // 2

    return np.array([[tx, ty], [bx, by], [lx, ly], [rx, ry], [cx, cy]], dtype='int32')


def generate_heatmap(points, h, w):
    """
    Arguments:
        points: a numpy int array with shape [num_points, 3].
            It is in the form [x, y, sigma],
            where sigma is the size of the gaussian peak.
            And it is assumed that `0 <= x < w` and `0 <= y < h`.
        h, w: integers.
    Returns:
        a numpy float array with shape [h, w].
    """

    k = 17  # sigmas mustn't be too large!
    a = np.arange(-k, k + 1, dtype='float32')
    X, Y = np.meshgrid(a, a)  # they have shape [2*k + 1, 2*k + 1]
    D = 0.5 * (X**2 + Y**2)
    heatmap = np.zeros([h + 2 * k, w + 2 * k], dtype='float32')

    for x, y, sigma in points:

        # point could be out of the image
        if 0 > x or x >= w or 0 > y or y >= h:
            continue

        g = np.exp(-D/(sigma**2))
        ymin, ymax = y, y + 2*k + 1
        xmin, xmax = x, x + 2*k + 1
        heatmap[ymin:ymax, xmin:xmax] = np.maximum(g, heatmap[ymin:ymax, xmin:xmax])

    return heatmap[k:-k, k:-k]


def random_crop(image_and_masks, extreme_points):

    height, width, _ = image_and_masks.shape
    min_dimension = min(height, width)
    size = np.random.randint(min_dimension // 2, min_dimension)
    x = np.random.randint(0, width - size)
    y = np.random.randint(0, height - size)

    ymin, xmin, ymax, xmax = y, x, y + size, x + size
    image_and_masks = image_and_masks[ymin:ymax, xmin:xmax]

    # `extreme_points` have coordinates
    # in ranges [0, width - 1] and [0, height - 1]
    extreme_points -= np.array([xmin, ymin])  # shape [num_boxes, 5, 2]

    # note that now some point coordinates might be out of image,
    # but they will be ignored when generating heatmaps
    return image_and_masks, extreme_points


def random_flip(image_and_masks, extreme_points):

    if np.random.rand() > 0.5:

        image_and_masks = cv2.flip(image_and_masks, 0)
        height, width, _ = image_and_masks.shape
        x, y = np.split(extreme_points, 2, axis=2)
        extreme_points = np.concat([width - x - 1, y], axis=2)

        # switch left and right
        correct_order = [0, 1, 3, 2, 4]
        extreme_points = extreme_points[:, correct_order]

    return image_and_masks, extreme_points


def random_color_jitter(image_and_masks):
    return image_and_masks
