import os, time, argparse
import numpy as np
from PIL import Image
import glob
import torch
from torchvision.utils import save_image as imwrite

from utils import load_checkpoint, tensor2cuda
from new.inceptionED3 import net


# 调用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def addTransparency(img, factor=1):
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img_blender, img, factor)
    return img


def main():
    # 开关定义

    parser = argparse.ArgumentParser(description ="network pytorch")


    parser.add_argument("--model", type=str, default="", help='checkpoint')
    parser.add_argument("--model_name", type=str, default='', help='model name')
    # value
    parser.add_argument("--intest", type=str, default="", help='input syn path')
    parser.add_argument("--outest", type=str, default=".", help='input syn path')
    argspar = parser.parse_args()


    print("\nnetwork pytorch")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()

    # train
    print('> Loading Generator...')
    name = arg.model_name
    Gmodel_name = name + '.tar'
    Dmodel_name = name + '.tar'
    G_Model, _= load_checkpoint(argspar.model, net, Gmodel_name)

    os.makedirs(arg.outest, exist_ok=True)
    test(argspar, G_Model)


def test(argspar, model):
    # init
    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x + 1) / 2
    files = os.listdir(argspar.intest)
    time_test = []
    model.eval()
    # test
    for i in range(len(files)):
        haze = np.array(Image.open(argspar.intest + '/' + files[i]).convert('RGB')) / 255

        with torch.no_grad():
            starttime = time.time()
            haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :]).cuda()
            haze = tensor2cuda(haze)
            haze = norm(haze)
            out  = model(haze)
            out = denorm(out)
            endtime1 = time.time()

            imwrite(out, argspar.outest + files[i], range=(0, 1))

            time_test.append(endtime1 - starttime)

            print('The ' + str(i) + ' Time: %.4f s.' % (endtime1 - starttime))

    print('Mean Time: %.4f s.' % (sum(time_test) / len(time_test)))



if __name__ == '__main__':
    main()
