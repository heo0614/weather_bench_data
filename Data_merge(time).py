import os
import re
import xarray as xr
import glob

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


base_dirs = [
        r"E:\metnet3\weather_bench\sparse_data(40)_input",
        r"E:\metnet3\weather_bench\dense_data_input",
        r"E:\metnet3\weather_bench\low_data_input",
        r"E:\metnet3\weather_bench\high_data_target",
        r"E:\metnet3\weather_bench\sparse_data(40)_target",
        r"E:\metnet3\weather_bench\dense_data_target"
        r"E:\metnet3\weather_bench\sparse_data_target"
    ]

for dir_path in base_dirs:
    merge_nc_files(dir_path)
