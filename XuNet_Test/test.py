"""This module is used to test the XuNet model."""
from glob import glob
import torch
import numpy as np
import imageio as io
from model.model import XuNet
from PIL import Image
from torchvision import transforms
TEST_BATCH_SIZE = 40
COVER_PATH = "../analysis_data_paper1/FNNS/cover/*.png"
STEGO_PATH = "../analysis_data_paper1/FNNS/stego/*.png"
CHKPT = "./checkpoints/net_437.pt"
resize=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
            ])
cover_image_names = glob(COVER_PATH)
stego_image_names = glob(STEGO_PATH)

cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

model = XuNet()

ckpt = torch.load(CHKPT,map_location='cpu')
model.load_state_dict(ckpt["model_state_dict"])
# pylint: disable=E1101
images = torch.empty((TEST_BATCH_SIZE, 1, 256, 256), dtype=torch.float)
# pylint: enable=E1101
test_accuracy = []

for idx in range(0, len(cover_image_names), TEST_BATCH_SIZE // 2):
    cover_batch = cover_image_names[idx : idx + TEST_BATCH_SIZE // 2]
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
        img=Image.open(batch[i])
        img=resize(img)
        # im=resize(torch.tensor(io.imread(batch[i],as_gray=True)))
        # print(im.shape)
        # print(img.shape)
        images[i, 0, :, :] = img.squeeze(0)
    image_tensor = images
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    # pylint: enable=E1101
    outputs = model(image_tensor,torch.device('cpu'))
    prediction = outputs.data.max(1)[1]

    accuracy = (
        prediction.eq(batch_labels.data).sum()
        * 100.0
        / (batch_labels.size()[0])
    )
    test_accuracy.append(accuracy.item())

print(f"test_accuracy = {sum(test_accuracy)/len(test_accuracy):%.2f}")
