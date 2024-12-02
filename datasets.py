from torch.utils import data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random

def remove_white_space_image(img_np: np.ndarray, padding: int =10):
    """
    获取白底图片中, 物体的bbox; 此处白底必须是纯白色.
    其中, 白底有两种表示方法, 分别是 1.0 以及 255; 在开始时进行检查并且匹配
    对最大值为255的图片进行操作.
    三通道的图无法直接使用255进行操作, 为了减小计算, 直接将三通道相加, 值为255*3的pix 认为是白底.
    :param img_np:
    :return:
    """
    # if np.max(img_np) <= 1.0:  # 1.0 <= 1.0 True
    #     img_np = (img_np * 255).astype("uint8")
    # else:
    #     img_np = img_np.astype("uint8")

    h, w, c = img_np.shape
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 300)  # max = 300
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[max(0, ymin - padding):min(h, ymax + padding), max(0, xmin - padding):min(w, xmax + padding),
                  :]
    return img_cropped
def resize_image_by_ratio(img_np: np.ndarray, size: int):
    """
    按照比例resize
    :param img_np:
    :param size:
    :return:
    """
    # print(len(img_np.shape))
    if len(img_np.shape) == 2:
        h, w = img_np.shape
    elif len(img_np.shape) == 3:
        h, w, _ = img_np.shape
    else:
        assert 0

    ratio = h / w
    if h > w:
        new_img = cv2.resize(img_np, (int(size / ratio), size,))  # resize is w, h  (fx, fy...)
    else:
        new_img = cv2.resize(img_np, (size, int(size * ratio),))
    # new_img[np.where(new_img < 200)] = 0
    return new_img

def make_img_square(img_np: np.ndarray):
    if len(img_np.shape) == 2:
        h, w = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1)) * np.max(img_np)
            white2 = np.ones((h, delta2)) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w)) * np.max(img_np)
            white2 = np.ones((delta2, w)) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((h, delta2, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((delta2, w, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img

class Datasets(data.Dataset):
    def __init__(self, root, datasets = "Sketchy",
                 img_txt = "Sketchy/1_img.txt", skt_txt = "Sketchy/1_skt_train.txt",
                 img_size=(224,224),length =10000):

        with open(img_txt, 'r') as f:
            img_path = f.readlines()

        with open(skt_txt, 'r') as f:
            skt_path = f.readlines()

        self.root = root
        self.img_size = img_size
        self.datasets = datasets
        self.length = length
        self.stage_class = {}
        for path in img_path:
            path = path[:-1]
            label = int(path.split(' ')[-1])
            path = path.rsplit(' ', 1)[0]

            if label not in self.stage_class.keys():
                self.stage_class[label]={"skt":[],"img":[]}

            self.stage_class[label]["img"].append(path)

        for path in skt_path:
            path = path[:-1]
            label = int(path.split(' ')[-1])
            path = path.rsplit(' ', 1)[0]

            self.stage_class[label]["skt"].append(path)

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])


        self.label = list(self.stage_class.keys())
        self.num_class = 1+max(self.label)

    def __len__(self):
        return self.length

    def get_negative(self,pos):
        neg = pos
        while(neg == pos):
            neg = random.choice(self.label)
        return  neg

    def __getitem__(self, idx):
        label = random.choice(self.label)

        img_path = self.root+self.datasets+"/"+random.choice(self.stage_class[label]["img"])
        neg_path = self.root+self.datasets+"/"+random.choice(self.stage_class[self.get_negative(label)]["img"])
        skt_path = self.root+self.datasets+"/"+random.choice(self.stage_class[label]["skt"])

        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.open(img_path).resize(self.img_size).convert('RGB')
        img = self.transform(img)

        neg = cv2.imread(neg_path)
        neg = cv2.resize(neg, self.img_size)
        neg = cv2.cvtColor(neg, cv2.COLOR_BGR2RGB)
        # neg = Image.open(neg_path).resize(self.img_size).convert('RGB')
        neg = self.transform(neg)

        skt = cv2.imread(skt_path)
        skt = cv2.cvtColor(skt, cv2.COLOR_BGR2RGB)
        skt = remove_white_space_image(skt)
        skt = resize_image_by_ratio(skt, self.img_size[0])
        skt = make_img_square(skt)
        skt = self.transform(skt)

        label = int(label)

        return img,skt,neg,label
class TestDatasets(data.Dataset):
    def __init__(self, root, datasets="Sketchy",stage = 5,img_size=(224, 224),m='skt'):

        self.path = []
        self.label = []
        label_acc = 0
        for i in range(1,stage+1):
            if m == "img":
                with open(f"datasets_list"+datasets+f"/{i}_img.txt", 'r') as f:
                    data =f.readlines()

                for item in data:
                    label = int(item[:-1].split(' ')[-1])+label_acc
                    path = item.rsplit(' ', 1)[0]
                    self.path.append(path)
                    self.label.append(label)

            else:
                with open(f"datasets_list"+datasets+f"/{i}_skt_test.txt", 'r') as f:
                    data = f.readlines()

                for item in data:
                    label = int(item[:-1].split(' ')[-1])+label_acc
                    path = item.rsplit(' ', 1)[0]
                    self.path.append(path)
                    self.label.append(label)
            label_acc = max(self.label)+1

        self.root = root
        self.img_size = img_size
        self.datasets = datasets
        self.m=m
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):return len(self.path)

    def __getitem__(self, idx):
        item = self.path[idx]
        label = self.label[idx]

        path = self.root + self.datasets + item

        if self.m == 'img':
            img = cv2.imread(path)
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            return img,  label

        else:
            skt = cv2.imread(path)
            skt = cv2.cvtColor(skt, cv2.COLOR_BGR2RGB)
            skt = remove_white_space_image(skt)
            skt = resize_image_by_ratio(skt, self.img_size[0])
            skt = make_img_square(skt)
            skt = self.transform(skt)

            return  skt, label
