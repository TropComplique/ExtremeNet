import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from detector.input_pipeline import ExtremePointsDataset
from detector.trainer import Trainer
from pycocotools.coco import COCO


NUM_EPOCHS = 24
BATCH_SIZE = 20
DEVICE = torch.device('cuda:0')

SAVE_PATH = 'models/run00.pth'
LOGS_DIR = 'summaries/run00/'
PRETRAINED_BACKBONE = 'pretrained/mobilenet.pth'

IMAGES = '/home/dan/datasets/COCO/images/train2017/'
ANNOTATIONS = '/home/dan/datasets/COCO/annotations/person_keypoints_train2017.json'
VAL_IMAGES = '/home/dan/datasets/COCO/images/val2017/'
VAL_ANNOTATIONS = '/home/dan/datasets/COCO/annotations/person_keypoints_val2017.json'


def train_and_evaluate():

    train = ExtremePointsDataset(
        COCO(ANNOTATIONS), image_folder=IMAGES,
        is_training=True, training_size=640
    )
    train_loader = DataLoader(
        dataset=train, batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True,
        shuffle=True, drop_last=True
    )
    val = ExtremePointsDataset(
        COCO(VAL_ANNOTATIONS),
        image_folder=VAL_IMAGES,
        is_training=False
    )
    val_loader = DataLoader(dataset=val, batch_size=1)

    num_steps = NUM_EPOCHS * (len(train) // BATCH_SIZE)
    model = Trainer(num_steps)
    state = torch.load(PRETRAINED_BACKBONE)
    model.network.backbone.load_state_dict(state)
    model.network.to(DEVICE)

    i = 0  # number of weight updates
    writer = SummaryWriter(LOGS_DIR)

    for e in range(1, NUM_EPOCHS + 1):

        model.network.train()
        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = {k: v.to(DEVICE) for k, v in labels.items()}
            losses = model.train_step(images, labels)

            for k, v in losses.items():
                writer.add_scalar(k, v.item(), i)

            print(f'epoch {e}, iteration {i}')
            i += 1

        eval_losses = []
        model.network.eval()
        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = {k: v.to(DEVICE) for k, v in labels.items()}
            losses = model.evaluate(images, labels)
            eval_losses.append({n: v.item() for n, v in losses.items()})

        eval_losses = {k: sum(d[k] for d in eval_losses)/len(eval_losses) for k in losses}
        for k, v in eval_losses.items():
            writer.add_scalar('eval_' + k, v, i)

        # save every epoch
        model.save(SAVE_PATH)


train_and_evaluate()
