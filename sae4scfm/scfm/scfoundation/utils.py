import numpy as np
import torch
from typing import Dict

from .scfoundation_model_files import select_model

def load_model_frommmf(best_ckpt_path, key='gene'):
    """Load model"""
    model_data = torch.load(best_ckpt_path,map_location='cpu')
    model_data = model_data[key]
    model_data = convertconfig(model_data)
    if not model_data.__contains__('config'):
        print('***** No config *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['config']
        print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None
    model = select_model(config)
    model_state_dict = model_data['model_state_dict']    
    model.load_state_dict(model_state_dict)
    return model.cuda(),config

def convertconfig(ckpt):
    newconfig = {}
    newconfig['config']={}
    model_type = ckpt['config']['model']
    
    for key, val in ckpt['config']['model_config'][model_type].items():
        newconfig['config'][key]=val
        
    for key, val in ckpt['config']['dataset_config']['rnaseq'].items():
        newconfig['config'][key]=val
        
    if model_type == 'performergau_resolution':
        model_type = 'performer_gau'
    
    import collections
    d = collections.OrderedDict()
    for key, val in ckpt['state_dict'].items():
        d[str(key).split('model.')[1]]=val
        
    newconfig['config']['model_type']=model_type
    newconfig['model_state_dict']=d
    newconfig['config']['pos_embed']=False
    newconfig['config']['device']='cuda'
    return newconfig

def gatherData(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    pad_id: int,
    pad_value: int
) -> Dict[str, torch.Tensor]:
    """Tokenize and pad gene expression data."""

    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )

    # First pass: find non-zero counts and determine actual max_len
    nonzero_counts = np.count_nonzero(data, axis=1)
    max_ori_len = nonzero_counts.max()
    max_len = min(max_ori_len, max_len)

    # Preallocate output tensors (avoids list appending and stacking)
    n_cells = data.shape[0]
    gene_ids_out = torch.full((n_cells, max_len), pad_id, dtype=torch.long)
    values_out = torch.full((n_cells, max_len), pad_value, dtype=torch.float32)

    # Single pass: extract non-zero values and populate tensors directly
    for i in range(n_cells):
        row = data[i]
        idx = np.nonzero(row)[0]
        n_nonzero = len(idx)
        
        if n_nonzero > max_len:
            # Sample max_len genes
            print('here', flush=True)
            sampled_idx = np.random.choice(n_nonzero, max_len, replace=False)
            idx = idx[sampled_idx]
            n_nonzero = max_len
        
        # Directly assign to preallocated tensors (no intermediate copies)
        gene_ids_out[i, :n_nonzero] = torch.from_numpy(gene_ids[idx])
        values_out[i, :n_nonzero] = torch.from_numpy(row[idx])

    batch_padded = {
        "genes": gene_ids_out,
        "values": values_out,
    }
    return batch_padded