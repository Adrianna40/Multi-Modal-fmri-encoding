import pandas as pd 
from pycocotools.coco import COCO
import os 
import numpy as np 
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import AvgPool1d
import torch 


def get_coco_info():
    nsd_stim_info = pd.read_csv(nsd_stim_info_file_path)
    nsd_stim_info = nsd_stim_info[nsd_stim_info[f'subject{subj}']==1]  
    nsd_to_coco = {nsd_id : coco_id for nsd_id, coco_id in zip(list(nsd_stim_info['nsdId']), list(nsd_stim_info['cocoId']))}
    coco_val=COCO(os.path.join(coco_annotation_path, 'captions_val2017.json'))
    coco_train=COCO(os.path.join(coco_annotation_path, 'captions_train2017.json'))
    coco = coco_val.anns
    coco.update(coco_train.anns)
    coco_id_to_description_all = {item['image_id']: item['caption'] for item in coco.values()}
    coco_id_to_description = {coco_id: coco_id_to_description_all[coco_id] for coco_id in nsd_stim_info['cocoId']}
    return nsd_to_coco, coco_id_to_description

def get_nsd_id(img_path):
    nsd_id_str = img_path.split('-')[-1][:-4]
    while nsd_id_str[0] == '0':
        nsd_id_str = nsd_id_str[1:]
    return int(nsd_id_str)
    
class TextDataset(Dataset):
    def __init__(self, imgs_paths, idxs):
        self.imgs_paths = np.array(imgs_paths)[idxs]

    def __len__(self):
        return len(self.imgs_paths)


    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        nsd_id = get_nsd_id(img_path)
        coco_id = nsd_to_coco[nsd_id]
        description = coco_id_to_description[coco_id]
        return description
    
def get_embeddings(paths, idxs, model, tokenizer):
    txt_dataset = TextDataset(paths, idxs)
    txt_loader = DataLoader(txt_dataset, batch_size=len(txt_dataset))
    tokenized_descriptions = tokenizer(next(iter(txt_loader)), padding='max_length', max_length=55, truncation = True, return_tensors="pt")
    dataset_tokens = TensorDataset(tokenized_descriptions['input_ids'], tokenized_descriptions['attention_mask'])
    dataloader_tokens = DataLoader(dataset_tokens, batch_size=100)
    model.to(device)
    max_tokens_len = tokenized_descriptions['input_ids'].shape[1]
    # mp = MaxPool1d(max_tokens_len)
    mp = AvgPool1d(max_tokens_len)
    hidden_states = []
    for batch in dataloader_tokens:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask, output_hidden_states=True)
            hs = outputs['hidden_states'][-1].cpu().detach()

            hidden_states.extend([mp(h.T).flatten().numpy() for h in hs])
    return np.array(hidden_states)