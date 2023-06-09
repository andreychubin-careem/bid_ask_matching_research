import numpy as np
import pandas as pd
from tqdm import tqdm
from pyhive import presto

from ._datagen import get_wusool_drivers
from ._data_operations import load_data
from ._setup import PRESETS


def get_snapshots(
        city: str,
        date: str = '2023-02-25',
        window_size: int = 3,
        save_dir: str = 'data'
) -> None:
    assert city in PRESETS.keys(), f'City is not present in presets. Available options are {list(PRESETS.keys())}'
    if window_size > 10:
        print(f'Window size > 10 is not well supported, which may lead to errors')
    PLACE = city
    AREA = PRESETS[city]['service_area_id']
    WINDOW = window_size
    TYPE = '' if WINDOW == 3 else f'_{WINDOW}min'

    conn = presto.connect(
        host='presto-python-r-script-cluster.careem-engineering.com',
        username='presto_python_r',
        port=8080
    )

    wusool_query = get_wusool_drivers(date, AREA)  # this does not works properly (no filtering at all)
    wusool = pd.read_sql(sql=wusool_query, con=conn)
    date_range = pd.date_range(start=f'{date} 00:00:00', end=f'{date} 23:50:00', freq='T')
    timesteps = np.random.choice(date_range, 100, replace=False)
    timesteps.sort()

    for time in tqdm(timesteps):
        stime = str(time).split('T')[1][:-6]
        etime = str(time + np.timedelta64(WINDOW, 'm')).split('T')[1][:-6]

        data = load_data(
            conn=conn,
            wusool=wusool,
            date=date,
            start_time=stime,
            end_time=etime,
            lat_bounds=PRESETS[city]['lat_bounds'],
            lon_bounds=PRESETS[city]['lon_bounds'],
            service_area_id=AREA,
            cut_off_km=200.0
        )

        data.to_parquet(f'{save_dir}/snapshots_{PLACE}{TYPE}/snapshot_{time}.pq')


if __name__ == '__main__':
    get_snapshots(city='jeddah', date='2023-03-01', window_size=3)
