from torch.utils.data import DataLoader
from detector.input_pipeline import ExtremePointsDataset
from detector.trainer import Trainer

import sys
sys.path.append('/home/dan/work/cocoapi/PythonAPI/')
from pycocotools.coco import COCO


NUM_STEPS = 50000
NUM_EPOCHS = 10
BATCH_SIZE = 8
PATH = 'models/run00.pth'
DEVICE = torch.device('cuda:0')
IMAGES = '/home/dan/datasets/COCO/images/train2017/'
ANNOTATIONS = '/home/dan/datasets/COCO/annotations/person_keypoints_train2017.json'
TRAIN_LOGS = ''


def train_and_evaluate():

    train = ExtremePointsDataset(
        COCO(ANNOTATIONS), image_folder=IMAGES,
        is_training=True, training_size=640
    )
    val = ExtremePointsDataset(
        COCO(ANNOTATIONS), image_folder=IMAGES,
        is_training=False
    )

    train_loader = DataLoader(
        dataset=train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True
    )

    model = Trainer(IMAGE_SIZE)
    model.network.to(DEVICE)

    i = 0
    logs = []
    text = 'e: {0}, i: {1}, total: {2:.3f}, heatmap: {3:.3f}, ' +\
           'offset: {4:.3f}, additional: {5:.3f}'

    for e in range(NUM_EPOCHS):

        model.network.train()
        for batch in train_loader:

            images, heatmaps, offsets, masks, num_boxes = batch
            images = images.to(DEVICE)
            labels = {
                'heatmaps': heatmaps.to(DEVICE), 'offsets': offsets.to(DEVICE),
                'masks': masks.to(DEVICE), 'num_boxes': num_boxes.to(DEVICE)
            }
            losses = model.train_step(images, labels)

            i += 1
            log = text.format(
                e, i, losses['total_loss'], losses['heatmap_loss'],
                losses['offset_loss'], losses['additional_loss']
            )
            print(log)
            logs.append(losses)

        model.network.eval()
        for batch in val_loader:

            images, heatmaps, offsets, masks, num_boxes = batch
            images = images.to(DEVICE)
            labels = {
                'heatmaps': heatmaps.to(DEVICE), 'offsets': offsets.to(DEVICE),
                'masks': masks.to(DEVICE), 'num_boxes': num_boxes.to(DEVICE)
            }
            losses = model.evaluate(images, labels)
            logs.append(losses)

        model.save(PATH)
        with open(TRAIN_LOGS, 'w') as f:
            json.dump(logs, f)


main()
