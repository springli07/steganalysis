"""This module is used to test the Srnet model."""
from glob import glob
import torch
import numpy as np
import imageio as io
from model.model import Srnet
import os

TEST_BATCH_SIZE = 2
COVER_PATH = "/home/u2108183005/Sr/data/cover/*"
STEGO_PATH = "/home/u2108183005/Sr/data/stego/*"
CHKPT = "./checkpoints/net_229.pt"
# print(os.listdir(COVER_PATH))
# for filename in os.listdir(COVER_PATH):
#     print(os.path.join(COVER_PATH,filename))
cover_image_names = glob(COVER_PATH)
# print(cover_image_names)
stego_image_names = glob(STEGO_PATH)

cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

model = Srnet().cuda()

ckpt = torch.load(CHKPT)
model.load_state_dict(ckpt["model_state_dict"])
# pylint: disable=E1101
images = torch.empty((TEST_BATCH_SIZE, 1, 256, 256), dtype=torch.float)
# pylint: enable=E1101
test_accuracy = []

for idx in range(0, len(cover_image_names), TEST_BATCH_SIZE // 2):
    cover_batch = cover_image_names[idx : idx + TEST_BATCH_SIZE // 2]
    # print(cover_batch)
    stego_batch = stego_image_names[idx : idx + TEST_BATCH_SIZE // 2]

    batch = []
    batch_labels = []

    xi = 0
    yi = 0
    for i in range(2 * len(cover_batch)):
        if i % 2 == 0:
            batch.append(stego_batch[xi])
            batch_labels.append(1)
            xi += 1
        else:
            batch.append(cover_batch[yi])
            batch_labels.append(0)
            yi += 1
    # pylint: disable=E1101
    for i in range(TEST_BATCH_SIZE):
        # print(batch[i])
        images[i, 0, :, :] = torch.tensor(io.imread(batch[i])).cuda()
    image_tensor = images.cuda()
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
    # pylint: enable=E1101
    outputs = model(image_tensor)
    prediction = outputs.data.max(1)[1]

    accuracy = (
        prediction.eq(batch_labels.data).sum()
        * 100.0
        / (batch_labels.size()[0])
    )
    test_accuracy.append(accuracy.item())
print(test_accuracy)
# print(f"test_accuracy = {sum(test_accuracy)/len(test_accuracy):%.2f}")
