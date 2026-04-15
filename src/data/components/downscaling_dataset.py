import os
import pandas as pd
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import zstandard
import io
import xarray as xr
from typing import Union, List


class DownscalingDataset(Dataset): 
    
    def __init__(self, dataset_dir: str, target_vars: dict, nn_lowres: bool = True,
                 static_vars: dict = None, crop_size: int = None,
                 metadata_file_name = 'metadata.csv',
                 skip_dynamic_load: bool = False):
        self.metadata_file_name = metadata_file_name
        self.dataset_dir = dataset_dir
        metadata_path = self.dataset_dir + self.metadata_file_name
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Dataset metadata not found: {metadata_path}\n"
                f"Expected directory (with metadata.csv) is usually 'LDM-downscaling/'. "
                f"Set paths.data_dir to that directory (e.g. .../LDM-downscaling/) in config or via "
                f"paths.data_dir=<your_repo>/LDM-downscaling when running."
            )
        self.metadata = pd.read_csv(metadata_path, parse_dates=['ref_time'])
        self.target_vars = target_vars
        self.nn_lowres = nn_lowres
        self.crop_size = crop_size
        self.skip_dynamic_load = skip_dynamic_load
        if static_vars:
            self.static_vars = static_vars
            self.static_data = {}
            for s_v in self.static_vars:
                if s_v == 'lc_tif_file':
                    tmp = self.get_LC_tif_data(self.static_vars[s_v])
                    for i, cat in enumerate(tmp):
                        self.static_data[s_v + '_cat_' + str(i+1)] = cat
                else:
                    # load static data
                    self.static_data[s_v] = self.get_tif_data(self.static_vars[s_v])
                    # normalize static data
                    self.static_data[s_v] = self.normalize(self.static_data[s_v])
        else:
            self.static_vars = static_vars
        self.res_list = ['low_res', 'high_res']

        if self.skip_dynamic_load:
            if not self.static_vars or self.nn_lowres:
                raise ValueError(
                    "skip_dynamic_load=True requires static_vars and nn_lowres=False "
                    "(Stage-1 static AE: skip decompressing dynamic .pt.zst per sample)."
                )
            probe = self.read_data(0)
            self._template_low = torch.zeros_like(self._stack_res_channels(probe, 'low_res'))
            self._template_high = torch.zeros_like(self._stack_res_channels(probe, 'high_res'))
            del probe

    def __len__(self) -> int:
        return len(self.metadata)

    def _stack_res_channels(self, data: dict, res_type: str) -> torch.Tensor:
        tensor_list = []
        for var in self.target_vars[res_type]:
            tensor = data[res_type][var]
            if res_type == 'low_res' and self.nn_lowres:
                tensor = torch.repeat_interleave(tensor, 8, dim=0)
                tensor = torch.repeat_interleave(tensor, 8, dim=1)
            if var == 'SST':
                tensor = torch.nan_to_num(tensor, nan=-10)
            tensor_list.append(tensor.to(torch.float32))
        if self.static_vars and self.nn_lowres and res_type == 'low_res':
            for s_v in self.static_data:
                tensor_list.append(self.static_data[s_v].to(torch.float32))
        return torch.stack(tensor_list)

    def _stack_static_tensors(self) -> torch.Tensor:
        tensor_list = []
        for s_v in self.static_data:
            tensor_list.append(self.static_data[s_v].to(torch.float32))
        return torch.stack(tensor_list)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor, int]:
        # MeanFlow Stage-1 (static_ctx AE): static fields are time-invariant; avoid loading
        # dynamic low/high zstd tensors every step — only HR static rasters + random crop.
        if self.skip_dynamic_load:
            results_dict = {
                'low_res': self._template_low,
                'high_res': self._template_high,
                'static': self._stack_static_tensors(),
            }
            if self.crop_size is not None:
                results_dict, _ = self.random_crop(results_dict, self.crop_size)
            return results_dict['static'], self.get_ref_time(idx)

        data = self.read_data(idx)
        results_dict = {}
        for res_type in self.res_list:
            results_dict[res_type] = self._stack_res_channels(data, res_type)

        # if you don't upscale LR but have static HRES vars, save them separetly! (==LDM conditioner)
        if self.static_vars and not self.nn_lowres:
            results_dict['static'] = self._stack_static_tensors()

        if self.crop_size is not None:
            results_dict, cropping_info_list = self.random_crop(results_dict, self.crop_size)

        if self.static_vars and not self.nn_lowres:
            return results_dict['low_res'], results_dict['high_res'], results_dict['static'], self.get_ref_time(idx)
            # if self.crop_size is not None:
            #     return results_dict['low_res'], results_dict['high_res'], results_dict['static'], self.get_ref_time(idx), cropping_info_list
            # else:
            #     return results_dict['low_res'], results_dict['high_res'], results_dict['static'], self.get_ref_time(idx)
        else:
            return results_dict['low_res'], results_dict['high_res'], self.get_ref_time(idx)

    def read_data(self, idx: int) -> dict:
        data = {}
        map_res_name = {'low_res': 'low',
                        'high_res': 'high'}
        try:
            for res_type in self.res_list:
                file_name = self.dataset_dir + self.metadata['files_path_' + map_res_name[res_type]][idx]
                if self.target_vars['high_res'] == ['2mT']:
                    file_name = file_name.replace('_high.pt.zst', '_high_2mT.pt.zst') 
                elif self.target_vars['high_res'] == ['U10', 'V10']:
                    file_name = file_name.replace('_high.pt.zst', '_high_UV.pt.zst') 
                elif self.target_vars['high_res'] == ['TP']:
                    file_name = file_name.replace('_high.pt.zst', '_high_TP.pt.zst') 
                with open(file_name, "rb") as f:
                    data_read = f.read()
                dctx = zstandard.ZstdDecompressor()
                decompressed = io.BytesIO(dctx.decompress(data_read))
                data[res_type] = torch.load(decompressed)[self.metadata['hour'][idx]]
        except:
            print('Cannot read data')
            raise
        return data

    def random_crop(self, batch: dict, crop_size = 128):
        try:
            ratio_hl = batch['high_res'].shape[-1]//batch['low_res'].shape[-1]
        except:
            ratio_hl = 1
        height, width = batch['high_res'].shape[-2:]
        if ratio_hl != 1:
            # if low_res is not upscaled: need to pick random top and left in low_res dimension 
            # and then convert them in h_res! otherwise would get disaligned low_res-high_res pairs!
            top = np.random.randint(0, height//ratio_hl - crop_size//ratio_hl + 1)
            left = np.random.randint(0, width//ratio_hl - crop_size//ratio_hl + 1)
            top = top*ratio_hl
            left = left*ratio_hl
        else:
            top = np.random.randint(0, height - crop_size + 1)
            left = np.random.randint(0, width - crop_size + 1)
        
        tops={'high_res': top, 'low_res': top//ratio_hl, 'static': top}
        lefts={'high_res': left, 'low_res': left//ratio_hl, 'static': left}
        crop_sizes={'high_res': crop_size, 'low_res': crop_size//ratio_hl, 'static': crop_size}

        if height < crop_size or width < crop_size:
            raise ValueError("Tensor size is smaller than crop size")

        for res_type in batch:
            batch[res_type] = batch[res_type][:, tops[res_type]:tops[res_type] + crop_sizes[res_type], lefts[res_type]:lefts[res_type] + crop_sizes[res_type]]

        return batch, {'tops':tops, 'lefts': lefts, 'crop_sizes': crop_sizes}
       
    def get_ref_time(self, idx: int) -> int:
        df_unix_sec = int(self.metadata['ref_time'][idx].timestamp() * 10**9) #  self.metadata['ref_time'][idx].astype(int)/ 10**9
        return df_unix_sec
    
    # def generate_metadata(self) -> pd.DataFrame:
    #     if not os.path.isfile(self.dataset_dir + self.metadata_file_name):
    #         print('Creating metadata...')
    #         high_list = sorted(glob(self.dataset_dir + '*/*_high.pt.zst'))
    #         low_list = sorted(glob(self.dataset_dir + '*/*_low.pt.zst'))
    #         meta_data={}
    #         for path_list in [high_list, low_list]:
    #             start_datetime = path_list[0].split('/')[-1].split('_')[0] + ' 00:00'
    #             end_datetime = path_list[-1].split('/')[-1].split('_')[0] + ' 23:00'
    #             res_type = path_list[0].split('/')[-1].split('_')[1].split('.')[0]
    #             col_name = 'files_path_' + res_type
    #             time_steps = pd.period_range(start=pd.to_datetime(start_datetime), 
    #                                          end=pd.to_datetime(end_datetime), freq='H').to_timestamp()
    #             meta_data[res_type] = pd.DataFrame({'ref_time': time_steps, 
    #                                       col_name: self.dataset_dir + 
    #                                       time_steps.strftime('%Y') + '/' +
    #                                       time_steps.strftime('%Y') + '-' + 
    #                                       time_steps.strftime('%m') + '-' + 
    #                                       time_steps.strftime('%d') + '_' + res_type + '.pt.zst'})
    #         merged_metadata = pd.merge(meta_data['low'], meta_data['high'], on="ref_time",
    #                                    how='inner').reset_index(drop=True)
    #         print(type(merged_metadata['ref_time']))
    #         merged_metadata['hour'] = merged_metadata['ref_time'].dt.hour
    #         merged_metadata.to_csv(self.dataset_dir + self.metadata_file_name, index=False)
    #         print('Done creating metadata...')
    #     else:
    #         merged_metadata = pd.read_csv(self.dataset_dir + self.metadata_file_name, parse_dates=['ref_time'])
    #     return merged_metadata

    def get_LC_tif_data(self, tif_file) -> torch.Tensor:
        tif_data = xr.open_dataset(tif_file, engine='rasterio')
        tif_data
        tensor_data_list = []
        for band in tif_data['band'].values:
            tensor_data_i = torch.from_numpy(tif_data['band_data'][band-1].values)
            tensor_data_list.append(tensor_data_i)
        tensor_data_all = torch.stack(tensor_data_list)
        return tensor_data_all

    def get_tif_data(self, tif_file) -> torch.Tensor:
        tif_data = xr.open_dataset(tif_file, engine='rasterio')
        tif_data = torch.from_numpy(tif_data['band_data'][0].values)
        return tif_data
    
    def normalize(self, tensor: torch.tensor):
        return (tensor - tensor.mean()) / tensor.std()
