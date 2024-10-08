import os
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# 처리할 파일들이 위치한 디렉토리 경로
raw_data_dir = r'E:\metnet3\weather_bench\raw_data'

# 결과물을 저장할 디렉토리 경로
sparse_input_dir = r'E:\metnet3\weather_bench\sparse_data_input'
dense_input_dir = r'E:\metnet3\weather_bench\dense_data_input'
low_input_dir = r'E:\metnet3\weather_bench\low_data_input'
high_data_target_dir = r'E:\metnet3\weather_bench\high_data_target'
sparse_target_dir = r'E:\metnet3\weather_bench\sparse_data_target'
dense_target_dir = r'E:\metnet3\weather_bench\dense_data_target'

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

#down sample : chat gpt
def downsample_spatial(ds, factor=2):

    return ds.coarsen(latitude=factor, longitude=factor, boundary='trim').mean()

#target size로 center crop
def center_crop(ds, target_size, dim1='latitude', dim2='longitude'):
    size1 = ds.dims[dim1]
    size2 = ds.dims[dim2]
    start1 = (size1 - target_size) // 2
    start2 = (size2 - target_size) // 2
    return ds.isel(latitude=slice(start1, start1 + target_size),
                   longitude=slice(start2, start2 + target_size))

#Level (고도) 에 따른 변수값 전체 평균
def average_over_level(ds):

    if 'level' in ds.dims:
        return ds.mean(dim='level')
    else:
        return ds


def process_files_grouped_by_date(nc_files):
    """
    동일한 날짜의 파일들을 그룹화하여 처리합니다.
    """
    # 파일명에서 날짜 정보를 추출하여 그룹화
    date_to_files = {}
    for file_path in nc_files:
        base_name = os.path.basename(file_path)
        parts = base_name.split('_')
        if len(parts) < 4:
            print(f"format {file_path}")
            continue
        date_str = parts[3]  # '2019-06-01'
        if date_str not in date_to_files:
            date_to_files[date_str] = []
        date_to_files[date_str].append(file_path)

    for date_str, files in tqdm(date_to_files.items(), desc="날짜별 파일 처리", unit="날짜"):
        datasets = []
        for file in files:
            ds = xr.open_dataset(file, decode_times=True)
            datasets.append(ds)

        if not datasets:
            print(f"날짜 {date_str} 없음")
            continue

        # 하나의 데이터셋으로 병합
        try:
            combined_ds = xr.concat(datasets, dim='time')
        except Exception as e:
            print(f"데이터셋을 병합하는 중 오류 발생: {date_str}, 오류: {e}")
            for ds in datasets:
                ds.close()
            continue

        try:
            # 624x624 데이터를 312x312로 다운샘플링 (raw_down)
            raw_down = downsample_spatial(combined_ds, factor=2)

            # raw_down을 중심 크롭하여 156x156으로
            cropped_156 = center_crop(raw_down, 156)

            # 3. sparse_data_input 
            sparse_vars = [
                '2m_temperature',
                'surface_pressure',
                'total_precipitation',
                'u_component_of_wind',
                'v_component_of_wind',
                '2m_dewpoint_temperature'
            ]
            sparse_data = cropped_156[sparse_vars]
            sparse_data = average_over_level(sparse_data)
            sparse_filename = f"156x156_sparse_0.5_input_{date_str}.nc"
            sparse_filepath = os.path.join(sparse_input_dir, sparse_filename)
            sparse_data.to_netcdf(sparse_filepath)

            # dense_data_input 
            dense_vars = [
                'geopotential',
                'land_sea_mask',
                'temperature',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'specific_humidity'
            ]
            dense_data = cropped_156[dense_vars]
            dense_data = average_over_level(dense_data)

            # 156x156_dense_0.5_input_날짜.nc
            dense_filename = f"156x156_dense_0.5_input_{date_str}.nc"
            dense_filepath = os.path.join(dense_input_dir, dense_filename)
            dense_data.to_netcdf(dense_filepath)

            # raw_down을 다시 다운샘플링하여 156x156으로 (low_data_input)
            low_down = downsample_spatial(raw_down, factor=2)
            low_vars = ['total_cloud_cover', 'total_precipitation']
            low_data = low_down[low_vars]
            low_data = average_over_level(low_data)

            # 156x156_low_1.0_input_날짜.nc
            low_filename = f"156x156_low_1.0_input_{date_str}.nc"
            low_filepath = os.path.join(low_input_dir, low_filename)
            low_data.to_netcdf(low_filepath)

            # raw_data에서 중심 크롭하여 64x64로, high_data_target
            cropped_64 = center_crop(combined_ds, 64)
            precip_data = cropped_64[['total_precipitation']]
            precip_data = average_over_level(precip_data)

            # 64x64_high_target_0.25_날짜.nc
            precip_filename = f"64x64_high_target_0.25_{date_str}.nc"
            precip_filepath = os.path.join(high_data_target_dir, precip_filename)
            precip_data.to_netcdf(precip_filepath)

            # sparse_data (156x156)를 다시 중심 크롭하여 32x32로
            cropped_32_sparse = center_crop(sparse_data, 32)

            # sparse_data_target 처리 ('2m_temperature', 'total_precipitation', '2m_dewpoint_temperature')
            sparse_target_vars = ['2m_temperature', 'total_precipitation', '2m_dewpoint_temperature']
            sparse_target = cropped_32_sparse[sparse_target_vars]

            # 32x32_sparse_target_0.5_날짜.nc
            sparse_target_filename = f"32x32_sparse_target_0.5_{date_str}.nc"
            sparse_target_filepath = os.path.join(sparse_target_dir, sparse_target_filename)
            sparse_target.to_netcdf(sparse_target_filepath)

            # dense_data_target
            cropped_32_dense = center_crop(dense_data, 32)

            # 10. dense_data_target 저장
            # 32x32_dense_target_0.5_날짜.nc
            dense_target_filename = f"32x32_dense_target_0.5_{date_str}.nc"
            dense_target_filepath = os.path.join(dense_target_dir, dense_target_filename)
            cropped_32_dense.to_netcdf(dense_target_filepath)

        except Exception as e:
            print(f"날짜 {date_str}의 파일 처리 중 오류 발생: {e}")

        finally:
            # 메모리 방지
            combined_ds.close()
            for ds in datasets:
                ds.close()



# raw_data 디렉토리 내 모든 .nc 파일 목록 가져오기
nc_files = glob(os.path.join(raw_data_dir, '*.nc'))

print(f"총 {len(nc_files)}개의 파일을 처리합니다.")

# 파일을 날짜별로 그룹화하여 처리
process_files_grouped_by_date(nc_files)

print("모든 파일 처리가 완료되었습니다.")



