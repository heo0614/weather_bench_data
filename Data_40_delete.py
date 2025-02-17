import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

from Data_division import sparse_input_dir_dense, sparse_target_dir_dense, data_root

parser = argparse.ArgumentParser()
parser.add_argument('--sparsity', type=int, default=40, help='percentage of data points to remove')
parser.add_argument('--seed', type=int, default=42, help='random seed')


def get_sparse_dirs(data_root, sparsity):
    return (data_root + 'sparse_data({})_input/'.format(sparsity),
            data_root + 'sparse_data({})_target/'.format(sparsity))


if __name__ == '__main__':
    args = parser.parse_args()
    # 랜덤 시드 설정
    np.random.seed(args.seed)

    # 지울 퍼센테이지
    sparsity = args.sparsity

    output_input_dir, output_target_dir = get_sparse_dirs(data_root, sparsity)

    # 출력 디렉토리가 존재하지 않으면 생성
    os.makedirs(output_input_dir, exist_ok=True)
    os.makedirs(output_target_dir, exist_ok=True)

    # 1. 마스킹 위치 생성 및 delete_position.csv 저장

    # sparse input 하나만 불러오기
    sample_input_file = os.path.join(sparse_input_dir_dense, '156x156_sparse_0.5_input_2019-06-01.nc')
    ds_input = xr.open_dataset(sample_input_file)

    # 위도와 경도 변수 추출
    # 위도, 경도
    if ('latitude' in ds_input.coords) and ('longitude' in ds_input.coords):
        lat = ds_input['latitude'].values
        lon = ds_input['longitude'].values
        if lat.ndim == 1 and lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
        elif lat.ndim == 2 and lon.ndim == 2:
            lat_grid = lat
            lon_grid = lon
        else:
            raise ValueError("lat, lon error input")
    else:
        raise ValueError("lat, lon error input")

    # 전체 그리드 포인트 수와 마스킹할 포인트 수 계산
    total_grid = lat_grid.shape[0] * lat_grid.shape[1]
    num_mask = int(sparsity * total_grid)

    # 그리드 인덱스를 평탄화하여 랜덤 선택
    flatten_indices = np.arange(total_grid)
    mask_indices = np.random.choice(flatten_indices, size=num_mask, replace=False)
    mask_2d_indices = np.unravel_index(mask_indices, lat_grid.shape)

    # 선택된 위치의 위도와 경도 추출
    masked_lat = lat_grid[mask_2d_indices]
    masked_lon = lon_grid[mask_2d_indices]

    # CSV 파일로 저장
    delete_position_csv = os.path.join(output_input_dir, 'delete_position.csv')
    df = pd.DataFrame({
        'latitude': masked_lat,
        'longitude': masked_lon
    })
    df.to_csv(delete_position_csv, index=False)
    print(f" {delete_position_csv} save.")

    # True = 마스킹할 위치
    mask = np.zeros(lat_grid.shape, dtype=bool)
    mask[mask_2d_indices] = True

    # netcdf 파일들
    input_files = [f for f in os.listdir(sparse_input_dir_dense) if f.endswith('.nc')]

    for file_name in input_files:
        input_path = os.path.join(sparse_input_dir_dense, file_name)
        ds = xr.open_dataset(input_path)

        # nan - var
        variables_to_mask = ['2m_temperature', 'surface_pressure', 'total_precipitation',
                             'u_component_of_wind', 'v_component_of_wind',
                             '2m_dewpoint_temperature']

        for var in variables_to_mask:
            if var in ds.data_vars:
                # lon, lat
                var_dims = ds[var].dims
                # lat lon check
                if ('latitude' in var_dims) and ('longitude' in var_dims):
                    # mask -> dataArray
                    mask_da = xr.DataArray(mask, coords={'latitude': ds['latitude'], 'longitude': ds['longitude']},
                                           dims=('latitude', 'longitude'))
                    # 마스크
                    ds[var] = ds[var].where(~mask_da)
                else:
                    print(f"{file_name}, {var}, diff - input.")
            else:
                print(f"{file_name}, {var}, diff - input.")

        # 파일명 _input -> _input(40del))
        if '_input' in file_name:
            new_file_name = file_name.replace('_input', '_input(40del)')
        else:
            new_file_name = file_name
            print(f"{file_name}, _input 포함안됨")

        output_path = os.path.join(output_input_dir, new_file_name)
        ds.to_netcdf(output_path)
        ds.close()
        print(f"{new_file_name} save")

    # 타겟 데이터
    sample_target_file = os.path.join(sparse_target_dir_dense, '128x128_sparse_target_0.5_2019-06-01.nc')
    ds_target = xr.open_dataset(sample_target_file)

    # 타겟 데이터의 위도와 경도 추출
    if ('latitude' in ds_target.coords) and ('longitude' in ds_target.coords):
        lat_t = ds_target['latitude'].values
        lon_t = ds_target['longitude'].values
        if lat_t.ndim == 1 and lon_t.ndim == 1:
            lon_t_grid, lat_t_grid = np.meshgrid(lon_t, lat_t)
        elif lat_t.ndim == 2 and lon_t.ndim == 2:
            lat_t_grid = lat_t
            lon_t_grid = lon_t
        else:
            raise ValueError("lat, lon error target")
    else:
        raise ValueError("lat, lon error target")

    # 타겟 데이터의 위도 및 경도 범위 계산
    lat_min = lat_t_grid.min()
    lat_max = lat_t_grid.max()
    lon_min = lon_t_grid.min()
    lon_max = lon_t_grid.max()

    # 마스크된 위치 중 타겟 범위 내에 있는지 확인
    within_lat = (masked_lat >= lat_min) & (masked_lat <= lat_max)
    within_lon = (masked_lon >= lon_min) & (masked_lon <= lon_max)
    within = within_lat & within_lon

    # 재조정된 마스크 인덱스
    adjusted_mask_indices = (mask_2d_indices[0][within], mask_2d_indices[1][within])

    # 재조정된 마스크 배열 생성
    adjusted_mask = np.zeros(lat_grid.shape, dtype=bool)
    adjusted_mask[adjusted_mask_indices] = True

    # target data load
    target_files = [f for f in os.listdir(sparse_target_dir_dense) if f.endswith('.nc')]

    for file_name in target_files:
        target_path = os.path.join(sparse_target_dir_dense, file_name)
        ds = xr.open_dataset(target_path)

        # 마스킹 변수
        variables_to_mask_target = ['2m_temperature', 'total_precipitation',
                                    '2m_dewpoint_temperature']

        for var in variables_to_mask_target:
            if var in ds.data_vars:
                # 타겟 데이터의 위도와 경도 추출
                if ('latitude' in ds.coords) and ('longitude' in ds.coords):
                    lat_target = ds['latitude'].values
                    lon_target = ds['longitude'].values
                    if lat_target.ndim == 1 and lon_target.ndim == 1:
                        lon_target_grid, lat_target_grid = np.meshgrid(lon_target, lat_target)
                    elif lat_target.ndim == 2 and lon_target.ndim == 2:
                        lat_target_grid = lat_target
                        lon_target_grid = lon_target
                    else:
                        print(f"{file_name}, {var}, diff - target.")
                        continue
                else:
                    print(f"{file_name}, {var}, diff - target.")
                    continue

                # 마스크된 위치 중 타겟 범위 내에 있는지 확인
                within_lat_t = (lat_target_grid >= lat_min) & (lat_target_grid <= lat_max)
                within_lon_t = (lon_target_grid >= lon_min) & (lon_target_grid <= lon_max)
                within_t = within_lat_t & within_lon_t

                # 타겟 그리드에 맞는 마스크 추출
                # 원래 마스크는 전체 그리드에 대한 것이므로 타겟 그리드에 맞게 중앙을 크롭
                y_total, x_total = lat_grid.shape
                y_target, x_target = lat_target_grid.shape
                start_y = (y_total - y_target) // 2
                start_x = (x_total - x_target) // 2
                end_y = start_y + y_target
                end_x = start_x + x_target

                # 마스크 크롭
                adjusted_mask_crop = adjusted_mask[start_y:end_y, start_x:end_x]

                if adjusted_mask_crop.shape != (y_target, x_target):
                    print(f"마스크 크기, 카겟 크기 다름. {file_name}")
                    continue

                # 타겟 그리드에 맞게 마스크 적용
                mask_da_target = xr.DataArray(adjusted_mask_crop,
                                              coords={'latitude': ds['latitude'], 'longitude': ds['longitude']},
                                              dims=('latitude', 'longitude'))
                ds[var] = ds[var].where(~mask_da_target)
            else:
                print(f"{file_name}, {var}")

        # 파일명 _target -> _target(40del))
        if '_target' in file_name:
            new_file_name = file_name.replace('_target', '_target(40del)')
        else:
            new_file_name = file_name
            print(f"{file_name} _target 포함안됨.")

        output_path = os.path.join(output_target_dir, new_file_name)
        ds.to_netcdf(output_path)
        ds.close()
        print(f"{new_file_name} save.")
