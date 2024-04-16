import cv2
import numpy as np

for i in range(1650):
    print(f'{i} / 1650')
    a = cv2.imread(f'./farmland_{i}.png')
    a = a.sum(axis=-1)
    h, w = a.shape
    b_0 = a.sum(axis=0)
    b_1 = a.sum(axis=1)
    top, bottom, left, right = 0, 0, 0, 0
    first = True
    for j in range(h):
        if b_1[j]:
            if first:
                top = j
            first = False
            bottom = j
    first = True
    for j in range(w):
        if b_0[j]:
            if first:
                left = j
            first = False
            right = j
    if i == 1649:
        print(f'{(h, w)}')
        print(f'{(top, bottom, left, right)}')
    b = a[top:bottom, left:right].astype(np.float32)
    b = cv2.resize(b, (400, 400), interpolation=cv2.INTER_CUBIC)
    b: np.ndarray = b != 0
    np.save(f'./farmland_{i}.npy', b)
    if i == 1649:
        print(b)
        print(b.shape)
