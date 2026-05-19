import cv2
import numpy as np
from PIL import Image



good = Image.open("./good.PNG")
good = np.array(good)
good = cv2.cvtColor(good, cv2.COLOR_RGB2HSV)


blank = cv2.imread("./blank.PNG")
blank = cv2.cvtColor(blank, cv2.COLOR_BGR2HSV)

#good = cv2.imread("good.PNG")
#good = cv2.cvtColor(good, cv2.COLOR_BGR2HSV)

some = cv2.imread("somewhat.PNG")
some = cv2.cvtColor(some, cv2.COLOR_BGR2HSV)


def accept(tile):

    lower_bound = np.array([90, 8, 103])
    upper_bound = np.array([180, 255, 255])

    mask = cv2.inRange(tile, lower_bound, upper_bound)
    
    ratio = np.count_nonzero(mask) / mask.size
    print("ratio", ratio)


accept(good)
accept(blank)
accept(some)

