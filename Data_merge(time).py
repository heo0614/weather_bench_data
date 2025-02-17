import argparse
import argparse
import glob
import os
import re

import xarray as xr

from Data_40_delete import get_sparse_dirs
from Data_division import dense_input_dir, dense_target_dir, low_input_dir, high_data_target_dir, \
    sparse_target_dir_dense, data_root

parser = argparse.ArgumentParser()
parser.add_argument('--sparsity', type=int, default=40, help='percentage of data points to remove')


# 파일명 날짜 제거
def get_base_filename(directory):
    nc_files = sorted(glob.glob(os.path.join(directory, '*.nc')))
    if not nc_files:
        print(f"파일없음: {directory}")
        return None

    sample_file = os.path.basename(nc_files[0])
    # YYYY-MM-DD
    base_name = re.sub(r'_\d{4}-\d{2}-\d{2}\.nc$', '', sample_file)
    return base_name


# NC 파일 불러오기
def merge_nc_files(directory):
    nc_files = sorted(glob.glob(os.path.join(directory, '*.nc')))

    if not nc_files:
        print(f"파일없음: {directory}")
        return

    # 기본 파일명 추출
    base_name = get_base_filename(directory)
    if not base_name:
        return

    # _all 추가
    output_filename = f"{base_name}_all.nc"
    output_path = os.path.join(directory, output_filename)

    # 'all.nc' 파일 있으면 스킵
    if os.path.exists(output_path):
        print(f"exist {output_path}\n")
        return

    # xarray의 open_mfdataset을 사용하여 여러 파일을 하나의 데이터셋으로 병합
    # combine='by_coords'는 좌표를 기준으로 병합
    try:
        ds = xr.open_mfdataset(nc_files, combine='by_coords')
        # 병합된 데이터셋을 'all.nc' 파일로 저장
        ds.to_netcdf(output_path)
        print(f"끝 {output_path}\n")
    except Exception as e:
        print(f"오류 {directory}: {e}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    sparsity = args.sparsity

    sparse_input_dir, sparse_target_dir = get_sparse_dirs(data_root, sparsity)

    if not os.path.exists(sparse_input_dir):
        print('Error: generate sparse data first using Data_40_delete.py!')
        exit()

    base_dirs = [
        sparse_input_dir,
        sparse_target_dir,
        dense_input_dir,
        dense_target_dir,
        low_input_dir,
        high_data_target_dir,
        sparse_target_dir_dense
    ]

    for dir_path in base_dirs:
        merge_nc_files(dir_path)
