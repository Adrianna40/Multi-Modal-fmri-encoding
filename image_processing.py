import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img).to(device)
        return img
    
def get_img_features(batch, feature_extractor):
    ft = feature_extractor(batch)
    ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
    return ft 

def fit_pca(feature_extractor, dataloader, n_components, batch_size):

    pca = IncrementalPCA(n_components, batch_size)
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        if len(d) >= pca.n_components:
            ft = get_img_features(d, feature_extractor).cpu().detach()
            pca.partial_fit(ft)
    return pca

def extract_features_with_pca(feature_extractor, dataloader, pca):
    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        ft = get_img_features(d, feature_extractor).cpu().detach()
        ft = pca.transform(ft)
        features.append(ft)
    return np.vstack(features)

def extract_features_no_pca(feature_extractor, dataloader):
    features = []
    for batch in dataloader:
        features.append(get_img_features(batch, feature_extractor).cpu().detach())
    return torch.vstack(features).numpy()
        
    