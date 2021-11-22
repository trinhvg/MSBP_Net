import cv2
import torch.utils.data as data
import data_load_colon as data_load
import data_load_prostate as data_load_p

####
class DatasetSerial(data.Dataset):
    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)
               
    def __init__(self, pair_list, shape_augs=None, input_augs=None):
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):

        pair = self.pair_list[idx]

        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1] # normal is 0

        # shape must be deterministic so it can be reused
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            input_img = shape_augs.augment_image(input_img)

        # additional augmentation just for the input
        if self.input_augs is not None:
            input_img = self.input_augs.augment_image(input_img)

        return input_img, int(img_label)
        
    def __len__(self):
        return len(self.pair_list)
    
####

def prepare_colon_tma_patch_data():
    return data_load.read_colon_dataset()

def prepare_prostate_tma_patch_data():
    return data_load_p.read_prostate_dataset()
