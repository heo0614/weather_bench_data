import argparse
import os

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# 데이터 저장 경로 설정
default_save_path = '../../datasets/weather_bench/'

parser = argparse.ArgumentParser()
parser.add_argument('--grid_size', type=int, default=624)
parser.add_argument('--save_path', type=str, default=default_save_path)
parser.add_argument('--start_date', type=str, default='2019-06-01')
parser.add_argument('--end_date', type=str, default='2019-07-31')


# 저장된 파일들의 크기를 확인하여 임계값 계산
# 해당 함수 만든 이유는 장시간 다운받다보면 멈추거 팅길 때, 중간에 멈춘 날짜는 다시 파일을 다운받아야하기 때문입니다.
# 임계값을 계산해서 해당 크기보다 낮으면 그 날짜부터 다운받습니다.
def calculate_file_size_threshold(save_path, threshold_factor=0.8, grid_size=512):
    file_sizes = []
    for date in dates:
        file_name = f'{save_path}{grid_size}x{grid_size}_0_25deg_{date.strftime("%Y-%m-%d")}_centered_on_Korea.nc'
        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name)
            file_sizes.append(file_size)

    if file_sizes:
        # 정상 파일 크기의 80%를 임계값으로 설정 (필요에 따라 수정 가능)
        avg_file_size = np.mean(file_sizes)
        return avg_file_size * threshold_factor
    else:
        # 기본적으로 최소 4GBMB 이하인 경우를 비정상으로 간주
        return 4 * 1024 * 1024 * 1024  # 4GBMB (adjust as needed)


# 위도와 경도 인덱스 범위 설정
def calculate_grid_indices(center_idx, half_count, max_length):
    start = max(0, center_idx - half_count)
    end = min(max_length, center_idx + half_count)
    actual_count = end - start
    if actual_count < grid_size:
        # Adjust start or end if at boundary
        if start == 0:
            end = min(max_length, start + grid_size)
        elif end == max_length:
            start = max(0, end - grid_size)
    return start, start + grid_size


if __name__ == '__main__':
    args = parser.parse_args()
    grid_size = args.grid_size
    save_path = args.save_path
    start_date = args.start_date
    end_date = args.end_date

    os.makedirs(save_path, exist_ok=True)

    # Google Cloud Storage 설정
    fs = gcsfs.GCSFileSystem(anon=True)

    # 0.25도 데이터 로드
    data_0_25 = xr.open_zarr(
        fs.get_mapper('gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'),
        consolidated=True)

    # 2달치 데이터 필터링 (예: 2019년 6월 1일부터 7월 31일까지)
    data_0_25 = data_0_25.sel(time=slice(start_date, end_date))

    for var in data_0_25.data_vars:
        print(f" - {var}")

    # 사용할 변수 목록
    variables_to_save = [
        # Sparse Input Variables
        '2m_temperature',
        'surface_pressure',
        'total_precipitation',
        'u_component_of_wind',
        'v_component_of_wind',
        '2m_dewpoint_temperature',

        # Dense Input Variables
        'geopotential',
        'land_sea_mask',
        'temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'specific_humidity',

        # Low Input Variables
        'total_cloud_cover',
        'sea_surface_temperature'
    ]

    # 변수를 선택해서 새로운 데이터셋 생성
    data_selected = data_0_25[variables_to_save]

    # 한국 중심 위도 경도 설정
    center_lat, center_lon = 36.5, 127

    # 위도와 경도에서 중심 좌표에 가장 가까운 그리드 셀 찾기
    lat_idx = np.abs(data_0_25.latitude - center_lat).argmin().values
    lon_idx = np.abs(data_0_25.longitude - center_lon).argmin().values

    # 원하는 그리드 크기 설정
    if grid_size % 2 != 0:
        # 이럴 경우에는 그냥 오류
        raise ValueError("number_of_grid must be an even integer.")

    half_grid_count = grid_size // 2  # 정수로 나눔

    lat_start, lat_end = calculate_grid_indices(lat_idx, half_grid_count, len(data_0_25.latitude))
    lon_start, lon_end = calculate_grid_indices(lon_idx, half_grid_count, len(data_0_25.longitude))

    # 데이터 형식을 float32로 변환하고 압축하여 저장
    compression_settings = dict(zlib=True, complevel=5)  # zlib 압축 사용, 압축 레벨 5
    encoding = {var: compression_settings for var in data_selected.data_vars}

    # 날짜별로 데이터를 나누어 저장
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # 임계값 설정
    min_file_size = calculate_file_size_threshold(save_path, grid_size=grid_size)

    for date in tqdm(dates, desc="Processing daily data"):
        # 날짜별 데이터 선택
        data_daily = data_selected.sel(time=str(date.date()))

        # 원하는 그리드 크기 데이터 추출
        data_grid_selected = data_daily.isel(latitude=slice(lat_start, lat_end), longitude=slice(lon_start, lon_end))

        # 파일 이름 지정 (동적으로 그리드 크기를 반영)
        file_name = f'{save_path}{grid_size}x{grid_size}_0_25deg_{date.strftime("%Y-%m-%d")}_centered_on_Korea.nc'

        # 파일이 이미 존재하는지 확인하고, 존재할 경우 크기 검사
        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name)
            # 파일이 정상적이지 않은 크기일 경우 삭제
            if file_size < min_file_size:
                os.remove(file_name)
                print(f"File {file_name} was too small ({file_size} bytes), so it was deleted.")
            else:
                print(f"File {file_name} already exists and is valid, skipping.")
                continue  # 이미 존재하고 정상적이라면 다음 날짜로 넘어감

        # 데이터 저장 (float32 타입 및 압축 적용)
        data_grid_selected.astype(np.float32).to_netcdf(file_name, encoding=encoding)
        print(f"File {file_name} saved successfully.")
