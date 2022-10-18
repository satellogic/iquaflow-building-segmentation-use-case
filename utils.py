import numpy as np
import albumentations as A
import h5py

def images_data_loader(folder_path):
    print("Loading images...")
    hfile = h5py.File(folder_path, 'r')
    hfile.keys()
    n1 = hfile.get('all_images')
    images = np.array(n1)
    print(images.shape)
    hfile.close()
    return images 

def masks_data_loader(folder_path):
    print("Loading masks...")
    hfile = h5py.File(folder_path, 'r')
    n1 = hfile.get('all_masks')
    masks = np.array(n1)
    print(masks.shape)
    print("Unique elements in the train mask:", np.unique(masks))
    hfile.close()
    return masks



transform = A.Compose([
    A.OneOf([
            A.HorizontalFlip(p=1),
            A.RandomRotate90(p=1),
            A.VerticalFlip(p=1)            
    ], p=1),
])



def data_augmentation(images, masks):
    print("Data augmentation step...")
    transformed_images = []
    transformed_masks = []
    for index, image in enumerate(images):
        transformed = transform(image=image, mask= masks[index])
        transformed_images.append(transformed['image'])
        transformed_masks.append(transformed['mask'])
        
    transformed_images = np.asarray(transformed_images)
    transformed_masks = np.asarray(transformed_masks)
    
    return transformed_images, transformed_masks