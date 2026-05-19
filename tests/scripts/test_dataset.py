from dinov2.data.datasets.slide_dataset import *
import cv2
import random

dataset = SlideDataset("/data/TCGA")



def hsv(tile_rgb, patch_size):

    tile = np.array(tile_rgb)
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    min_ratio = .6

    lower_bound = np.array([90, 8, 103])
    upper_bound = np.array([180, 255, 255])

    mask = cv2.inRange(tile, lower_bound, upper_bound)

    ratio = np.count_nonzero(mask) / mask.size
    if ratio > min_ratio:
            #print("accept this")
            #tile_rgb.show()
        return tile_rgb
    else:
            #tile_rgb.show()
        return None


i = 0
finish = 3072 * 1000000

tries = 0
for e in range(0, finish):
    for i in range(0, dataset.__len__()):
        #item = dataset.__getitem__(i)
        image, path = dataset.get_all(i)
    
        patch_size = 224
        #radomly pick level, read region
        for level in range(0, image.level_count):
        
            height = image.level_dimensions[0][1]
            width = image.level_dimensions[0][0]
            
            tries = 0
            while True:

                tries += 1   
                x = random.randint(0, width - patch_size)
                y = random.randint(0, height - patch_size)

                print("tying", tries)
                patch = image.read_region((x, y), level = level, size=(224, 224))
                res = hsv(patch, (224,224))
                if res == None:
                    if tries == 1000:
                        break
                    else:
                        continue
                else:
                    print(path, x, y, level, flush = True)
                    break
                



print("done")
