import os
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# 처리할 파일들이 위치한 디렉토리 경로
raw_data_dir = r'/projects/aiid/KIPOT_SKT/Weather/raw_data/'

# 결과물을 저장할 디렉토리 경로
sparse_input_dir = r'/projects/aiid/KIPOT_SKT/Weather/sparse_data_input'
dense_input_dir = r'/projects/aiid/KIPOT_SKT/Weather/dense_data_input'
low_input_dir = r'/projects/aiid/KIPOT_SKT/Weather/low_data_input'
high_data_target_dir = r'/projects/aiid/KIPOT_SKT/Weather/high_data_target'
sparse_target_dir = r'/projects/aiid/KIPOT_SKT/Weather/sparse_data_target'
dense_target_dir = r'/projects/aiid/KIPOT_SKT/Weather/dense_data_target'

# 저장 디렉토리가 없으면 생성
output_dirs = [
    sparse_input_dir,
    dense_input_dir,
    low_input_dir,
    high_data_target_dir,
    sparse_target_dir,
    dense_target_dir
]
for directory in output_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

def downsample_spatial(ds, factor=2):
    # 지정한 factor로 다운샘플링(평균)
    return ds.coarsen(latitude=factor, longitude=factor, boundary='trim').mean()

def center_crop(ds, target_size, dim1='latitude', dim2='longitude'):
    # target_size로 중심부분 잘라내기
    size1 = ds.dims[dim1]
    size2 = ds.dims[dim2]
    start1 = (size1 - target_size) // 2
    start2 = (size2 - target_size) // 2
    return ds.isel(**{dim1: slice(start1, start1 + target_size),
                      dim2: slice(start2, start2 + target_size)})

# raw_data 디렉토리 내 모든 .nc 파일 목록 가져오기
nc_files = glob(os.path.join(raw_data_dir, '*.nc'))
print(f"총 {len(nc_files)}개의 파일을 처리합니다.")

# 전체 파일을 한 번에 불러오기
# 이 때 combine='by_coords'로 좌표에 따라 자동 병합
ds = xr.open_mfdataset(nc_files, combine='by_coords', decode_times=True)

# 원 데이터 2배 다운샘플링(예: 624x624 -> 312x312)
raw_down = downsample_spatial(ds, factor=2)

# 156x156으로 center crop
cropped_156 = center_crop(raw_down, 156)

# sparse_data_input (156x156)
sparse_vars = [
    '2m_temperature',
    'surface_pressure',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    '2m_dewpoint_temperature'
]
sparse_data = cropped_156[sparse_vars]

# dense_data_input (156x156)
dense_vars = [
    'geopotential',
    'land_sea_mask',
    'total_precipitation',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'specific_humidity'
]
dense_data = cropped_156[dense_vars]

# low_data_input (156x156): raw_down에서 다시 2배 다운샘플링 -> 156x156
low_down = downsample_spatial(raw_down, factor=2)
low_vars = ['total_cloud_cover', 'total_precipitation']
low_data = low_down[low_vars]

# high_data_target (128x128): 원본 ds에서 바로 크롭
cropped_64 = center_crop(ds, 128)
precip_data = cropped_64[['total_precipitation']]

# sparse_data_target (32x32): sparse_data에서 크롭
cropped_32_sparse = center_crop(sparse_data, 32)
sparse_target_vars = ['2m_temperature', 'temperature', '2m_dewpoint_temperature']
sparse_target_data = cropped_32_sparse[sparse_target_vars]

# dense_data_target (32x32): dense_data에서 크롭
cropped_32_dense = center_crop(dense_data, 32)

# 최종 결과를 하나의 파일로 저장
def save_dataset(ds, out_dir, filename):
    out_path = os.path.join(out_dir, filename)
    ds.to_netcdf(out_path)
    print(f"{out_path} 저장 완료")

save_dataset(sparse_data, sparse_input_dir, "156x156_sparse_0.5_input_all(channel).nc")
save_dataset(dense_data, dense_input_dir, "156x156_dense_0.5_input_all(channel).nc")
save_dataset(low_data, low_input_dir, "156x156_low_1.0_input_all(channel).nc")
save_dataset(precip_data, high_data_target_dir, "128x128_high_target_0.25_all(channel).nc")
save_dataset(sparse_target_data, sparse_target_dir, "32x32_sparse_target_0.5_all(channel).nc")
save_dataset(cropped_32_dense, dense_target_dir, "32x32_dense_target_0.5_all(channel).nc")

print("전체 기간 처리 및 파일 저장이 완료되었습니다.")
