# coding=utf-8
import torch
import torchvision.models as models
import cv2

IMG_PATH = "/home/wangbin/PycharmProjects/quantizednn/data/smurf.jpeg"


def image_read(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).div(255).sub(0.5).float()
    img = torch.unsqueeze(img, 0)
    return img


def torch_modules(model_):
    print("module.modules()\n")
    for e in model_.modules():
        print(type(e), e)

    print("modules._modules.keys()\n")
    for e in model_._modules.keys():
        print(type(e), e)

    print("modules.children.keys()\n")
    for e in model_.children():
        print(type(e), e)


if __name__ == "__main__":
    image = image_read(IMG_PATH)
    model = models.resnet18(pretrained=True)
    model = torch.nn.DataParallel(model)
    model.eval()

    pred = model(image)
    print(pred.size())
