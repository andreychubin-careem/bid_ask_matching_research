import warnings
import shapely
import pandas as pd
from pyhive import presto
from pyproj import Geod

from ._datagen import get_drivers_location_snapshot, get_radar_calls


warnings.filterwarnings('ignore')
geod = Geod(ellps="WGS84")


def get_distance(line: shapely.LineString) -> float:
    return geod.geometry_length(line)/1000.0


def preprocess(data: pd.DataFrame, id_column: str) -> pd.DataFrame:
    frame = data.copy()
    frame = frame.sort_values('ts').drop_duplicates(subset=[id_column], keep='first')
    frame = frame[(~frame['latitude'].isna()) & (~frame['longitude'].isna())]
    frame['latitude'] = round(frame['latitude'].astype(float), 7)
    frame['longitude'] = round(frame['longitude'].astype(float), 7)
    frame['point'] = frame.apply(lambda row: shapely.Point((row['longitude'], row['latitude'])), axis=1)
    frame = frame.drop(['ts', 'longitude', 'latitude'], axis=1)
    frame['key'] = 1
    
    return frame


def to_matrix(customers: pd.DataFrame, drivers: pd.DataFrame, cut_off_km: float) -> pd.DataFrame:
    matrix = customers.merge(drivers, on='key', how='outer').drop('key', axis=1)
    matrix['line'] = matrix.apply(lambda row: shapely.LineString([row['point_x'], row['point_y']]), axis=1)
    matrix['distance'] = matrix['line'].apply(get_distance)
    matrix = matrix.drop(['point_x', 'point_y', 'line'], axis=1)
    matrix = matrix[matrix.distance <= cut_off_km]
    
    return matrix


def load_data(
    conn: presto.Connection,
    wusool: pd.DataFrame,
    date: str = '2023-02-15',
    start_time: str = '17:00:00.000',
    end_time: str = '17:01:00.000',
    lat_bounds: tuple = (24.82, 25.32),
    lon_bounds: tuple = (54.93, 55.65),
    service_area_id: int = 1,
    cut_off_km: float = 100.0
) -> pd.DataFrame:
    drivers_query = get_drivers_location_snapshot(
        date=date,
        start_time=start_time,
        end_time=end_time,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds
    )

    drivers = pd.read_sql(sql=drivers_query, con=conn)
    drivers = drivers[~drivers['driver_id'].isin(wusool['driver_id'])]
    drivers = preprocess(drivers, 'driver_id')
    
    customers_query = get_radar_calls(
        date=date,
        service_area_id=service_area_id,
        start_time=start_time,
        end_time=end_time
    )

    customers = pd.read_sql(sql=customers_query, con=conn) 
    customers = preprocess(customers, 'userid')
    
    matrix = to_matrix(customers, drivers, cut_off_km)
    
    return matrix
    