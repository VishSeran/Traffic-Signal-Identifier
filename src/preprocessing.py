import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

IMG_SIZE = (32,32)

def _readImg(path:str)-> np.ndarray:
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Failed to read an image: {path}")
    img = cv2.resize(img,IMG_SIZE,interpolation=cv2.INTER_AREA)
    img = apply_img_processing(img)
    return img

def apply_img_processing(img):

    #convert to HSV for color-based enhancement
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #Histogram equilization on v channel
    hsv[:,:,2] =cv2.equalizeHist(hsv[:,:,2])

    #Gaussian blur to remove noise
    img = cv2.GaussianBlur(img,(3,3),0)

    #convert to gray scale and back
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    return img

def load_train_set(data_root: str = 'dataset/Train'):
    X,y = [],[]

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Train directory not found: {data_root}")

    class_dirs = sorted(
        [
            d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,d))
        ]
    ) 

    for cls in class_dirs:
        cls_dirs = os.path.join(data_root,cls)
        label = int(cls)

        for fname in os.listdir(cls_dirs):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".ppm")):
                try:
                    img = _readImg(os.path.join(cls_dirs,fname))
                    X.append(img)
                    y.append(label)
                
                except Exception as e:
                    print(f"[warn] {e}")
    

    return np.array(X,dtype=np.uint8), np.array(y,dtype=np.int64)


def train_val_split(X, y,val_size=0.2,seed=42):
    X = X.astype(np.float32)/255.0
    return train_test_split(X,y,train_size=val_size, random_state=seed,stratify=y)
