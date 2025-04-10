import json
import math
import os
import random
import numpy as np

import cv2

"""
目录结构：
root:
    /bg   存放背景图片
    /img  存放标注的文件 labelme标注的语义分割文件  jpg+json
    /images 用于保存合成的图片
    /labels 保存yolo格式的标签文件

"""


def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    a = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1))
    if a < 0:
        a = 0
    b = (min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
    if b < 0:
        b = 0
    inter = a * b
    # Intersection area
    # inter = np.clip((min(b1_x2, b2_x2) - max(b1_x1, b2_x1)), 0) * \
    #         np.clip((min(b1_y2, b2_y2) - max(b1_y1, b2_y1)), 0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    area = min(w1 * h1, w2 * h2)

    iou = inter / area
    return iou  # IoU


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im, degrees=10, translate=.0, scale=.0, shear=0., perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    return M, im


def readfiles(p, exts=['jpg', 'jpeg', 'png']):
    L = []
    for root, dir, files in os.walk(p):
        for f in files:
            if f.split('.')[-1].lower() in exts:
                L.append(os.path.join(root, f))
    return L


if __name__ == '__main__':
    # 文件主目录
    root = r'F:\dataset\bird\20231012\hc'
    # 合成数据总数量
    total = 2000
    # 每次合成使用多少标签文件
    max_images = 3
    # 背景文件路径
    bg_path = os.path.join(root, 'bg')
    bgs = readfiles(bg_path)
    # 标注文件路径
    bz_path = os.path.join(root, 'img')
    bz_images = readfiles(bz_path)

    # 生成文件的起始序号
    count = 0
    for i in range(total):
        # 随机获取一个背景图片
        bg = random.choice(bgs)
        bg_img = cv2.imread(bg)
        H, W = bg_img.shape[:2]

        # 保存生成的文件路径
        save_images = os.path.join(root, 'images', str(count).rjust(8, '0') + '.jpg')
        save_txt = os.path.join(root, 'labels', str(count).rjust(8, '0') + '.txt')
        f = open(save_txt, 'a')
        L = []
        for j in range(max_images):
            # 随机获取一个标注文件
            bz_image = random.choice(bz_images)
            print(bz_image)
            bz1 = cv2.imread(bz_image)
            shape1 = bz1.shape

            bz_label = bz_image.replace('.jpg', '.json')
            if not os.path.exists(bz_label):
                continue
            try:
                data = json.load(open(bz_label, 'r'))
            except:
                continue
            shapes = data['shapes']

            # 读取json标签文件 根据变换矩阵更新标签坐标
            for shape in shapes:
                # 语义分割背景版
                mask = np.zeros(bg_img.shape, dtype=np.int32)
                bz = bz1.copy()
                # 对标注文件透视变换
                M, bz = random_perspective(bz)
                radio_h = random.uniform(0.2, 0.9)
                radio_w = random.uniform(0.2, 0.9)
                H_, W_ = int(H * radio_h), int(W * radio_w)
                top_ = (random.randint(0, W - W_), random.randint(0, H - H_))  # 新的top顶点

                label = shape['label']
                points = shape['points']
                points = np.array(points)
                xy = np.ones((points.shape[0], 3))
                xy[:, :2] = points
                xy = xy @ M.T
                points = xy[:, :2]
                xmin = min(points[:, 0])
                xmax = max(points[:, 0])
                ymin = min(points[:, 1])
                ymax = max(points[:, 1])
                if xmin <= 0 or ymin <= 0 or xmax >= shape1[1] or ymax >= shape1[0]:
                    continue
                # 对标注文件按比例缩放
                bz, ratio, pad = letterbox(bz, (H_, W_), auto=False, scaleup=True)
                # cv2.imshow('1', bz)
                # cv2.waitKey(0)
                mask1 = mask.copy()
                mask1[top_[1]:top_[1] + H_, top_[0]:top_[0] + W_] = bz
                # cv2.imshow('1', mask1.astype(np.uint8))
                # cv2.waitKey(0)
                points[:, 0] = ratio[0] * points[:, 0] + pad[0]  # top left x
                points[:, 1] = ratio[1] * points[:, 1] + pad[1]  # top left x
                points = points.astype(np.int32)
                xmin = min(points[:, 0])
                xmax = max(points[:, 0])
                ymin = min(points[:, 1])
                ymax = max(points[:, 1])
                if xmin <= 0 or ymin <= 0 or xmax >= W_ or ymax >= H_:
                    continue
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2
                w = (xmax - xmin)
                h = (ymax - ymin)
                if h * w < 64:
                    continue
                points[:, 0] = points[:, 0] + top_[0]
                points[:, 1] = points[:, 1] + top_[1]
                x = x + top_[0]
                y = y + top_[1]
                target = True
                for k in L:
                    iou = bbox_iou(np.array([x, y, w, h]), k)
                    if iou > 0.5:
                        target = False
                        break
                if not target:
                    continue
                L.append(np.array([x, y, w, h]))
                yolo_txt = ' '.join([label, str(x / W), str(y / H), str(w / W), str(h / H)])
                f.write(yolo_txt + '\n')
                # bz = cv2.fillConvexPoly(bz, points, (0, 255, 0), 0)

                mask = cv2.fillConvexPoly(mask, points, (1, 1, 1), 0)
                bg_img = np.where(mask, mask1.astype(np.uint8), bg_img)
        cv2.imwrite(save_images, bg_img)
        # cv2.imshow('a', bg_img)
        # cv2.imshow('b', bz)
        # cv2.waitKey(0)
        f.close()
        count += 1
