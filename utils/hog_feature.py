# Ref: https://www.cnblogs.com/Jack-Elvis/p/11285290.html
import cv2
import numpy as np


def L2Norm(cells):
    block = cells.flatten().astype(np.float32)
    norm_factor = np.sqrt(np.sum(block**2) + 1e-6)
    block /= norm_factor
    return block


def calc_hist(mag, angle, bin_size=9):
    hist = np.zeros((bin_size, ), dtype=np.int32)
    bin_step = 180 // bin_size
    bins = (angle // bin_step).flatten()
    flat_mag = mag.flatten()
    for i, m in zip(bins, flat_mag):
        hist[i] += m
    return hist


def calc_hog(gray, block_size = 2, cell_size = 2, bin_size = 9, eps = 1e-3):
    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    angle = np.int32(np.arctan(dy / (dx + eps)) * 180 / np.pi) + 90
    dx = cv2.convertScaleAbs(dx)
    dy = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)

    img_h, img_w = gray.shape[:2]
    cell_h, cell_w = (img_h // cell_size, img_w // cell_size)

    cells = np.zeros((cell_h, cell_w, bin_size), dtype = np.int32)
    for i in range(cell_h):
        cell_row = cell_size * i
        for j in range(cell_w):
            cell_col = cell_size * j
            cells[i, j] = calc_hist(
                mag[cell_row : cell_row + cell_size, cell_col : cell_col + cell_size], 
                angle[cell_row : cell_row + cell_size, cell_col : cell_col + cell_size], 
                bin_size
            )

    block_size = 2
    block_h, block_w = (cell_h - block_size + 1, cell_w - block_size + 1)
    blocks = np.zeros((block_h, block_w, block_size * block_size * bin_size), dtype = np.float32)
    for i in range(block_h):
        for j in range(block_w):
            blocks[i, j] = L2Norm(cells[i : i + block_size, j : j + block_size])

    return blocks.flatten()


def calc_hog_feature_size(img_size = 32, block_size = 2, cell_size = 2, bin_size = 9):
    assert img_size % cell_size == 0 and 360 % bin_size == 0
    block_feature = cell_size * cell_size * bin_size
    block_num = (img_size / cell_size - block_size + 1) ** 2
    return block_num * block_feature
