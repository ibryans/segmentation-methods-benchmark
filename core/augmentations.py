import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps, ImageChops
# from matplotlib import pyplot as plt


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        mask = Image.fromarray(mask, mode='L')
        if len(img.shape) == 2:
            img = Image.fromarray(img, mode=None) 
            assert img.size == mask.size
            for a in self.augmentations:
                img, mask = a(img, mask)
            return np.array(img), np.array(mask, dtype=np.uint8)
        elif len(img.shape) == 3:
            img = Image.fromarray(img.transpose(1,2,0), mode='RGB') 
            assert img.size == mask.size
            for a in self.augmentations:
                img, mask = a(img, mask)
            return np.array(img).transpose(2,0,1), np.array(mask, dtype=np.uint8)
        else: 
            raise RuntimeError(f'There is no implementation for image dimension {img.shape}')


class AddNoise(object):
    def __call__(self, img, mask):
        if img.im.bands == 1:
            noise = np.random.normal(loc=0,scale=0.02,size=(img.size[1], img.size[0]))
        else:
            noise = np.random.normal(loc=0,scale=0.02,size=(img.size[1], img.size[0], img.im.bands))
        return img + noise, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            # NOTE: we use FLIP_TOP_BOTTOM here intentionaly. Due to the dimensions of the image, it ends up being a horizontal flip.
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask
    

class RandomVerticallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask
   

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size), Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))

    
class RandomRotate(object):
    def __init__(self, degree=5.):
        self.degree = degree
        # self.counter = 0

    def __call__(self, img, mask):
        '''
        PIL automatically adds zeros to the borders of images that rotated. To fix this 
        issue, the code in the botton sets anywhere in the labels (mask) that is zero to 
        255 (the value used for ignore_index).
        '''
        rotate_degree = random.random() * 2 * self.degree - self.degree
        
        # plt.figure(dpi=90, figsize=(10,10))
        # plt.imshow(np.asarray(mask), cmap='tab20')
        # plt.colorbar();
        # plt.savefig(f'runs/plots/aut_{self.counter}')
        # plt.close()

        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask =  mask.rotate(rotate_degree, Image.NEAREST)

        binary_mask = Image.fromarray(np.ones([mask.size[1], mask.size[0]]))
        binary_mask = binary_mask.rotate(rotate_degree, Image.NEAREST)
        binary_mask = np.array(binary_mask)

        mask_arr = np.array(mask)
        mask_arr[binary_mask==0] = 255
        mask = Image.fromarray(mask_arr)
        
        # plt.figure(dpi=90, figsize=(10,10))
        # plt.imshow(np.asarray(mask), cmap='tab20')
        # plt.colorbar();
        # plt.savefig(f'runs/plots/aug_{self.counter}')
        # plt.close()

        # self.counter += 1
        return img, mask


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))