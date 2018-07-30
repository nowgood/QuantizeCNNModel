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


def my_hook(m, i, o):
    fm[0] = (i[0].data.clone())
    fm[1] = (o.data.clone())
    print('m:', type(m))
    print('i:', type(i))
    print('len(i):', len(i))
    print('i[0]:', type(i[0]))
    print('i[0]:', i[0].size())
    print('o:', type(o))
    print()
    print('i[0] shape:', i[0].size())
    print('o shape:', o.size())


def my_hook2(m, i, o):
    m.register_buffer("layer3", i[0])
    m.register_buffer("layer4", o)


if __name__ == "__main__":
    image = image_read(IMG_PATH)
    model = models.resnet18(pretrained=True)
    last = model._modules.get("layer4")
    fm = [0, 0]
    hook = last.register_forward_hook(my_hook2)
    model = torch.nn.DataParallel(model)
    model.eval()
    pred = model(image)
    print(model)
    for k, v in model._modules.items():
        print(k, v)

    hook.remove()