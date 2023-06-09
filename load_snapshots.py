import numpy as np
import pandas as pd
from tqdm import tqdm
from pyhive import presto

from datagen import get_wusool_drivers
from data_operations import load_data


if __name__ == '__main__':
    PLACE = 'riyadh' # 'riyadh', 3 | 'jeddah', 5 | dubai, 1
    AREA = 3
    WINDOW = 10 # 3
    TYPE = f'_{WINDOW}min' # '_10min'
    
    conn = presto.connect(
        host='presto-python-r-script-cluster.careem-engineering.com',
        username='presto_python_r',
        port=8080
    )
    
    date = '2023-02-26'
    
    wusool_query = get_wusool_drivers(date, AREA)
    wusool = pd.read_sql(sql=wusool_query, con=conn)
    
    date_range = pd.date_range(start=f'{date} 00:00:00', end=f'{date} 23:40:00', freq='T')
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
            lat_bounds=(24.00, 26.50), # (24.00, 26.50) | (21.05, 22.00) | (24.82, 25.32)
            lon_bounds=(43.95, 47.50), # (43.95, 47.50) | (38.90, 39.55) | (54.93, 55.65)
            service_area_id=AREA,
            cut_off_km=200.0
        )

        data.to_parquet(f'data/snapshots_{PLACE}{TYPE}/snapshot_{time}.pq')
