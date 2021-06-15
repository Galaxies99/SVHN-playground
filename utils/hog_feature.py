import cv2
import numpy as np


class HOG_Feature_Descriptor(object):
    def __init__(self, image_size = (32, 32), pixels_per_cell = (2, 2), cells_per_block = (2, 2), bin_size = 9, eps = 1e-6, debug = False):
        super(HOG_Feature_Descriptor, self).__init__()
        self.image_size = image_size
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.bin_size = bin_size
        self.debug = debug
        assert 180 % self.bin_size == 0
        assert image_size[0] % pixels_per_cell[0] == 0 and image_size[1] % pixels_per_cell[1] == 0
        self.cell_H = image_size[0] // pixels_per_cell[0]
        self.cell_W = image_size[1] // pixels_per_cell[1]
        self.block_H = self.cell_H - cells_per_block[0] + 1
        self.block_W = self.cell_W - cells_per_block[1] + 1
        self.eps = eps


    @staticmethod
    def calc_hist(maginitudes, angles, bin_size):
        mags = maginitudes.flatten()
        angs = angles.flatten()
        bin_step = 180 // bin_size
        res = np.zeros(bin_size, dtype = np.float)
        for mag, ang in zip(mags, angs):
            ang_left = int(ang // bin_step)
            ang_right = ang_left + 1
            mag_left = mag * (ang_right * bin_step - ang) / bin_step
            mag_right = mag * (ang - ang_left * bin_step) / bin_step
            if ang_left == bin_size:
                ang_left = 0
                ang_right = 1
            if ang_right == bin_size:
                ang_right = 0
            res[ang_left] += mag_left
            res[ang_right] += mag_right
        return res
    

    @staticmethod
    def L2_normalization(cells, eps = 1e-6):
        res = cells.flatten().astype(np.float32)
        norm = np.sqrt(np.sum(res ** 2) + eps)
        return res / norm
        
    
    def RGB_extractor(self, img):
        '''
        Single RGB Image Extractor: RGB -> HOG
        Input RGB should be in the shape of H * W * C, where C = 3, and H, W is specified in image_size.
        '''
        H, W, C = img.shape
        if H != self.image_size[0] or W != self.image_size[1] or C != 3:
            raise AttributeError('The size of the image mismatches with the given shape.')

        img_rightshifted = np.concatenate((img[:, 1:, :], np.zeros((H, 1, C))), axis = 1)
        img_leftshifted = np.concatenate((np.zeros((H, 1, C)), img[:, 0: W - 1, :]), axis = 1)
        horizontal_grad = img_rightshifted - img_leftshifted
        
        img_downshifted = np.concatenate((img[1:, :, :], np.zeros((1, W, C))), axis = 0)
        img_upshifted = np.concatenate((np.zeros((1, W, C)), img[0: H - 1, :, :]), axis = 0)
        vertical_grad = img_downshifted - img_upshifted
        
        magnitudes = horizontal_grad ** 2 + vertical_grad ** 2
        angles = np.arctan(horizontal_grad / (vertical_grad + self.eps)) * 180 / np.pi + 90
        zipped_array = np.stack((magnitudes, angles), axis = -1)
        zipped_array = np.max(zipped_array, axis = 2)
        magnitudes = np.sqrt(zipped_array[:, :, 0])
        angles = zipped_array[:, :, 1]

        cells = np.zeros((self.cell_H, self.cell_W, self.bin_size), dtype = np.float)
        for i in range(self.cell_H):
            for j in range(self.cell_W):
                begin_h = i * self.pixels_per_cell[0]
                begin_w = j * self.pixels_per_cell[1]
                cells[i, j, :] = self.calc_hist(
                    magnitudes[begin_h: begin_h + self.pixels_per_cell[0], begin_w: begin_w + self.pixels_per_cell[1]],
                    angles[begin_h: begin_h + self.pixels_per_cell[0], begin_w: begin_w + self.pixels_per_cell[1]],
                    self.bin_size
                )
        
        features = np.zeros((self.block_H, self.block_W, self.cells_per_block[0] * self.cells_per_block[1] * self.bin_size))
        for i in range(self.block_H):
            for j in range(self.block_W):
                features[i, j, :] = self.L2_normalization(cells[i: i + self.cells_per_block[0], j: j + self.cells_per_block[1], :])
        
        return features.flatten()


    def gray_extractor(self, img):
        '''
        Single Gray Image Extractor: Gray -> HOG
        Input Gray should be in the shape of H * W, where H, W is specified in image_size.
        '''
        H, W = img.shape
        if H != self.image_size[0] or W != self.image_size[1]:
            raise AttributeError('The size of the image mismatches with the given shape.')

        img_rightshifted = np.concatenate((img[:, 1:], np.zeros((H, 1))), axis = 1)
        img_leftshifted = np.concatenate((np.zeros((H, 1)), img[:, 0: W - 1]), axis = 1)
        horizontal_grad = img_rightshifted - img_leftshifted
        
        img_downshifted = np.concatenate((img[1:, :], np.zeros((1, W))), axis = 0)
        img_upshifted = np.concatenate((np.zeros((1, W)), img[0: H - 1]), axis = 0)
        vertical_grad = img_downshifted - img_upshifted
        
        magnitudes = np.sqrt(horizontal_grad ** 2 + vertical_grad ** 2)
        angles = np.arctan(horizontal_grad / (vertical_grad + self.eps)) * 180 / np.pi + 90

        cells = np.zeros((self.cell_H, self.cell_W, self.bin_size), dtype = np.float)
        for i in range(self.cell_H):
            for j in range(self.cell_W):
                begin_h = i * self.pixels_per_cell[0]
                begin_w = j * self.pixels_per_cell[1]
                cells[i, j, :] = self.calc_hist(
                    magnitudes[begin_h: begin_h + self.pixels_per_cell[0], begin_w: begin_w + self.pixels_per_cell[1]],
                    angles[begin_h: begin_h + self.pixels_per_cell[0], begin_w: begin_w + self.pixels_per_cell[1]],
                    self.bin_size
                )
        
        features = np.zeros((self.block_H, self.block_W, self.cells_per_block[0] * self.cells_per_block[1] * self.bin_size))
        for i in range(self.block_H):
            for j in range(self.block_W):
                features[i, j, :] = self.L2_normalization(cells[i: i + self.cells_per_block[0], j: j + self.cells_per_block[1], :])
        
        return features.flatten()
    

    def batch_RGB_extractor(self, img):
        '''
        Batch RGB Image Extractor: RGB -> HOG
        Input RGB should be in the shape of B * H * W * C, where C = 3, and H, W is specified in image_size.
        '''
        B, H, W, C = img.shape
        if H != self.image_size[0] or W != self.image_size[1] or C != 3:
            raise AttributeError('The size of the image mismatches with the given shape.')

        img_rightshifted = np.concatenate((img[:, :, 1:, :], np.zeros((B, H, 1, C))), axis = 2)
        img_leftshifted = np.concatenate((np.zeros((B, H, 1, C)), img[:, :, 0: W - 1, :]), axis = 2)
        horizontal_grad = img_rightshifted - img_leftshifted
        
        img_downshifted = np.concatenate((img[:, 1:, :, :], np.zeros((B, 1, W, C))), axis = 1)
        img_upshifted = np.concatenate((np.zeros((B, 1, W, C)), img[:, 0: H - 1, :, :]), axis = 1)
        vertical_grad = img_downshifted - img_upshifted
        
        magnitudes = horizontal_grad ** 2 + vertical_grad ** 2
        angles = np.arctan(horizontal_grad / (vertical_grad + self.eps)) * 180 / np.pi + 90
        zipped_array = np.stack((magnitudes, angles), axis = -1)
        zipped_array = np.max(zipped_array, axis = 3)
        magnitudes = np.sqrt(zipped_array[:, :, :, 0])
        angles = zipped_array[:, :, :, 1]

        cells = np.zeros((B, self.cell_H, self.cell_W, self.bin_size), dtype = np.float)
        for b in range(B):
            for i in range(self.cell_H):
                for j in range(self.cell_W):
                    begin_h = i * self.pixels_per_cell[0]
                    begin_w = j * self.pixels_per_cell[1]
                    cells[b, i, j, :] = self.calc_hist(
                        magnitudes[b, begin_h: begin_h + self.pixels_per_cell[0], begin_w: begin_w + self.pixels_per_cell[1]],
                        angles[b, begin_h: begin_h + self.pixels_per_cell[0], begin_w: begin_w + self.pixels_per_cell[1]],
                        self.bin_size
                    )
        
        features = np.zeros((B, self.block_H, self.block_W, self.cells_per_block[0] * self.cells_per_block[1] * self.bin_size))
        for b in range(B):
            for i in range(self.block_H):
                for j in range(self.block_W):
                    features[b, i, j, :] = self.L2_normalization(cells[b, i: i + self.cells_per_block[0], j: j + self.cells_per_block[1], :])
            
        return features.reshape(B, -1)
    
    def batch_gray_extractor(self, img):
        '''
        Batch Gray Image Extractor: Gray -> HOG
        Input Gray should be in the shape of B * H * W, where H, W is specified in image_size.
        '''
        B, H, W = img.shape
        if H != self.image_size[0] or W != self.image_size[1]:
            raise AttributeError('The size of the image mismatches with the given shape.')

        img_rightshifted = np.concatenate((img[:, :, 1:], np.zeros((B, H, 1))), axis = 2)
        img_leftshifted = np.concatenate((np.zeros((B, H, 1)), img[:, :, 0: W - 1]), axis = 2)
        horizontal_grad = img_rightshifted - img_leftshifted
        
        img_downshifted = np.concatenate((img[:, 1:, :], np.zeros((B, 1, W))), axis = 1)
        img_upshifted = np.concatenate((np.zeros((B, 1, W)), img[:, 0: H - 1, :]), axis = 1)
        vertical_grad = img_downshifted - img_upshifted
        
        magnitudes = np.sqrt(horizontal_grad ** 2 + vertical_grad ** 2)
        angles = np.arctan(horizontal_grad / (vertical_grad + self.eps)) * 180 / np.pi + 90

        cells = np.zeros((B, self.cell_H, self.cell_W, self.bin_size), dtype = np.float)
        for b in range(B):
            for i in range(self.cell_H):
                for j in range(self.cell_W):
                    begin_h = i * self.pixels_per_cell[0]
                    begin_w = j * self.pixels_per_cell[1]
                    cells[b, i, j, :] = self.calc_hist(
                        magnitudes[b, begin_h: begin_h + self.pixels_per_cell[0], begin_w: begin_w + self.pixels_per_cell[1]],
                        angles[b, begin_h: begin_h + self.pixels_per_cell[0], begin_w: begin_w + self.pixels_per_cell[1]],
                        self.bin_size
                    )
        
        features = np.zeros((B, self.block_H, self.block_W, self.cells_per_block[0] * self.cells_per_block[1] * self.bin_size))
        for b in range(B):
            for i in range(self.block_H):
                for j in range(self.block_W):
                    features[b, i, j, :] = self.L2_normalization(cells[b, i: i + self.cells_per_block[0], j: j + self.cells_per_block[1], :])
            
        return features.reshape(B, -1)
