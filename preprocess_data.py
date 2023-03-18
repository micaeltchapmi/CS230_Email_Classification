import os
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import random
import albumentations as A

data_dir = "./data/images"
N = 11026
resize_dim = [100,  100]

def count_files_in_directory():
    count = 0
    for dir in os.listdir(data_dir):
        count += len(os.listdir(osp.join(data_dir, dir)))
    print("total number of files : %d" % (count))

def get_min_dims():
    count = 1
    min_area = 1e10
    min_height_width = []
    for d in os.listdir(data_dir):
        files = os.listdir(osp.join(data_dir, d))
        for f in files:
            print("processing %d / %d" % (count, N))
            count+=1
            img = cv2.imread(osp.join(data_dir, d, f))
            W,H,C = img.shape
            cur_area = W * H
            if cur_area < min_area:
                min_height_width = [W, H]
                min_area = cur_area

def plot_images(imgs, n_row=1, n_col=1):
    _, axs = plt.subplots(n_row, n_col, figsize=(9, 6))
    axs = axs.flat
    for img, ax in zip(imgs, axs):
        ax.imshow(img, interpolation='nearest', aspect='auto')
    
    # one liner to remove *all axes in all subplots*
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    
    plt.show()

def resize(img, W=100, H=100):
    resized_image = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return resized_image  

def resize_images():
    count = 1
    for d in os.listdir(data_dir):
        files = os.listdir(osp.join(data_dir, d))
        for f in files:
            print("processing %d / %d" % (count, N))
            count+=1
            img = cv2.imread(osp.join(data_dir, d, f))
            resized_img = resize(img)
            plot_images([img, resized_img], 1, 2)

def save_images(imgs, split, d):
    count = 0
    outdir = "./data/EC_Data/%s/%s" % (split, d)
    os.makedirs(outdir, exist_ok=True)

    for img in imgs:
        fpath = osp.join(outdir, "%s_%05d.png" % (d, count))
        cv2.imwrite(fpath, img)
        count+=1

def split_and_save_images(imgs, d):

    train, val, test = [], [], []
    n = len(imgs)
    train = imgs[0 : int(0.7*n)]
    val = imgs[int(0.7*n) : int(0.9*n)]
    test = imgs[int(0.9*n) : ]

    save_images(train, "train", d)
    save_images(val, "val", d)
    save_images(test, "test", d)

def augment(img_files, d):
    # blur, horizontal flip, gaussian noise, brightness, contrast

    transform_fn = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.5),
        A.AdvancedBlur(p=0.5)
    ])

    augmented_images = []

    # read and resize current images
    for f in img_files:
        img = cv2.imread(osp.join(data_dir, d, f))
        resized_img = resize(img)
        augmented_images.append(resized_img)

    # augment to obtain 500 images
    random_indices = list(range(len(img_files)))
    random.shuffle(random_indices)
    n = len(img_files)
    i = 0
    while len(augmented_images) < 500:
        # read random img in directory
        file_indx = random_indices[i % n]
        img = cv2.imread(osp.join(data_dir, d, img_files[file_indx]))
        resized_img = resize(img)
        
        # augment image with a random combination of augmentation methods  
        augmented_img = transform_fn(image=resized_img)["image"]

        augmented_images.append(augmented_img)

        i += 1
    
    return augmented_images

def augment_images():
    count = 1
    N = len(os.listdir(data_dir))
    for d in os.listdir(data_dir):
        print("processing dir %d / %d" % (count, N))
        files = os.listdir(osp.join(data_dir, d))

        #augment the images in this directory
        augmented_images = augment(files, d)

        # save the augmented images to disk
        split_and_save_images(augmented_images, d)
        count += 1

def show_samples():
    count = 1
    N = len(os.listdir(data_dir))

    all_imgs = []

    for d in os.listdir(data_dir)[0:5]:
        print("processing dir %d / %d" % (count, N))
        files = os.listdir(osp.join(data_dir, d))
        count+=1

        # read and resize current images
        i = 0
        for f in files:
            img = cv2.imread(osp.join(data_dir, d, f))
            resized_img = resize(img)
            all_imgs.append(resized_img)
            i+=1
            if i == 5:
                break
    
    plot_images(all_imgs, n_row=5, n_col=5)

def augment_test(img_path):
    # blur, horizontal flip, gaussian noise, brightness, contrast

    transform_fn = A.Compose([
        #A.RandomBrightnessContrast(p=1.0),
        #A.HorizontalFlip(p=1.0),
        A.GaussNoise(p=1.0),
        #A.AdvancedBlur(p=1.0)
    ])

    img = cv2.cvtColor(cv2.imread(img_path, 3), cv2.COLOR_BGR2RGB)
    # augment image with a random combination of augmentation methods
    resized_img = resize(img)  
    augmented_img = transform_fn(image=resized_img)["image"]

    plt.imshow(augmented_img)
    plt.axis("off")
    plt.show()


def main():
    augment_images()

if __name__=="__main__":
    main()
