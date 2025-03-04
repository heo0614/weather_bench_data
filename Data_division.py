import os
from glob import glob
import xarray as xr
import numpy as np
from tqdm import tqdm
from Data_downloader_large import data_root, raw_data_dir

# -------------------------------------------------
# Define Sparse, Dense, Low variables
# -------------------------------------------------

sparse_vars = [
    '2m_temperature',
    'surface_pressure',
    'total_precipitation',
    'u_component_of_wind',
    'v_component_of_wind',
    '2m_dewpoint_temperature'
]
dense_vars = [
    'geopotential',
    'land_sea_mask',
    'temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'specific_humidity'
]
low_vars = [
    'total_cloud_cover',
    'total_precipitation'
]
sparse_target_vars = [
    '2m_temperature',
    'total_precipitation',
    '2m_dewpoint_temperature'
]
dense_target_vars = dense_vars
high_target_vars = ['total_precipitation']

# 결과물을 저장할 디렉토리 경로
sparse_input_dir_dense = data_root + 'sparse_data_input/'
dense_input_dir = data_root + 'dense_data_input/'
low_input_dir = data_root + 'low_data_input/'
high_data_target_dir = data_root + 'high_data_target/'
sparse_target_dir_dense = data_root + 'sparse_data_target/'
dense_target_dir = data_root + 'dense_data_target/'

# Define size of data
input_res = 156
high_target_res = 128
target_res = 32

# down sample : chat gpt
def downsample_spatial(ds, factor=2):
    return ds.coarsen(latitude=factor, longitude=factor, boundary='trim').mean()

def upsample_spatial(ds, factor=2):
    return ds.interp(
        latitude=np.linspace(ds.latitude.min(), ds.latitude.max(), ds.latitude.size * factor),
        longitude=np.linspace(ds.longitude.min(), ds.longitude.max(), ds.longitude.size * factor)
    )


# target size로 center crop
def center_crop(ds, target_size, dim1='latitude', dim2='longitude'):
    # target_size로 중심부분 잘라내기
    size1 = ds.dims[dim1]
    size2 = ds.dims[dim2]
    start1 = (size1 - target_size) // 2
    start2 = (size2 - target_size) // 2
    return ds.isel(**{dim1: slice(start1, start1 + target_size),
                      dim2: slice(start2, start2 + target_size)})

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
        if len(datasets) == 1:
            combined_ds = datasets[0]
        else:
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
            # 312x312로 다운샘플링 data를 다시 다운샘플링하여 156x156으로 (low_down)
            low_down = downsample_spatial(raw_down, factor=2)
            # raw_down을 중심 크롭하여 156x156으로
            cropped_156 = center_crop(raw_down, input_res)
            
            #1. Input Process
            
            # sparse_data_input
            sparse_data = cropped_156[sparse_vars]
            sparse_filename = f"{input_res}x{input_res}_sparse_0.5_input_{date_str}.nc"
            sparse_filepath = os.path.join(sparse_input_dir_dense, sparse_filename)
            sparse_data.to_netcdf(sparse_filepath)

            # dense_data_input
            dense_data = cropped_156[dense_vars]
            dense_filename = f"{input_res}x{input_res}_dense_0.5_input_{date_str}.nc"
            dense_filepath = os.path.join(dense_input_dir, dense_filename)
            dense_data.to_netcdf(dense_filepath)

            # low_data_input
            low_data = low_down[low_vars]
            low_filename = f"{input_res}x{input_res}_low_1.0_input_{date_str}.nc"
            low_filepath = os.path.join(low_input_dir, low_filename)
            low_data.to_netcdf(low_filepath)

            #2. Target Process
            
            # sparse_data (156x156)를 다시 중심 크롭하여 target_res로
            cropped_target_sparse = center_crop(sparse_data, target_res)
            # dense_data (156x156)를 다시 중심 크롭하여 target_res로
            cropped_target_dense = center_crop(dense_data, target_res)
            # high_target (128x128) 생성
            cropped_high_target = center_crop(combined_ds, high_target_res/2) # => 64crop
            upsampled_high_target = upsample_spatial(cropped_high_target, factor = 2) #=> 128x128 resolution
            
            # sparse_data_target
            sparse_target = cropped_target_sparse[sparse_target_vars]
            sparse_target_filename = f"{target_res}x{target_res}_sparse_target_0.5_{date_str}.nc"
            sparse_target_filepath = os.path.join(sparse_target_dir_dense, sparse_target_filename)
            sparse_target.to_netcdf(sparse_target_filepath)

            # dense_data_target
            dense_target = cropped_target_dense[dense_target_vars]
            dense_target_filename = f"{target_res}x{target_res}_dense_target_0.5_{date_str}.nc"
            dense_target_filepath = os.path.join(dense_target_dir, dense_target_filename)
            dense_target.to_netcdf(dense_target_filepath)
            
            # high_data_target
            high_target = upsampled_high_target[high_target_vars]
            high_target_filename = f"{high_target_res}x{high_target_res}_high_target_0.25_{date_str}.nc"
            high_target_filepath = os.path.join(high_data_target_dir, high_target_filename)
            high_target.to_netcdf(high_target_filepath)



        except Exception as e:
            print(f"날짜 {date_str}의 파일 처리 중 오류 발생: {e}")

        finally:
            # 메모리 방지
            combined_ds.close()
            for ds in datasets:
                ds.close()


if __name__ == '__main__':
    # 저장 디렉토리가 없으면 생성
    output_dirs = [
        sparse_input_dir_dense,
        dense_input_dir,
        low_input_dir,
        high_data_target_dir,
        sparse_target_dir_dense,
        dense_target_dir
    ]
    for directory in output_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # raw_data 디렉토리 내 모든 .nc 파일 목록 가져오기
    nc_files = glob(os.path.join(raw_data_dir, '*.nc'))

    print(f"총 {len(nc_files)}개의 파일을 처리합니다.")

    # 파일을 날짜별로 그룹화하여 처리
    process_files_grouped_by_date(nc_files)

    print("모든 파일 처리가 완료되었습니다.")
