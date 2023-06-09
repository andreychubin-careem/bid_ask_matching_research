def get_drivers_location_snapshot(
    date: str,
    start_time: str = '17:00:00.000',
    end_time: str = '17:01:00.000',
    lat_bounds: tuple = (24.82, 25.32),
    lon_bounds: tuple = (54.93, 55.65)
) -> str:
    return f"""
        select
            cast(driver_id as varchar) as driver_id,
            latitude,
            longitude,
            concat('{date} ', cast(date_add('millisecond', adma_location_read_at, TIME '00:00:00.000') as varchar)) as ts
        from prod_stg.driver_ping_parquet
        where 1=1
            and day = date('{date}')
            and latitude between {lat_bounds[0]} and {lat_bounds[1]}
            and longitude between {lon_bounds[0]} and {lon_bounds[1]}
            and status = 1
            and date_add('millisecond', adma_location_read_at, TIME '00:00:00.000') between TIME '{start_time}' and TIME '{end_time}'
            and not blocked_flag
    """


def get_radar_calls(
    date: str,
    service_area_id: int = 1,
    start_time: str = '17:00:00.000',
    end_time: str = '17:01:00.000'
) -> str:
    return f"""
        with ios_radar_calls as (
            SELECT
                concat('i', userid) as userid,
                concat('{date} ', cast(date_add('millisecond', cast(timestamp as bigint), TIME '00:00:00.000') as varchar)) as ts,
                latitude,
                longitude
            FROM app_events.icma
            WHERE date = '{date}'
                and eventname = 'radar_call'
                and service_area_id = '{service_area_id}'
                and date_add('millisecond', cast(timestamp as bigint), TIME '00:00:00.000') between TIME '{start_time}' and TIME '{end_time}'
        ),
        ----------------------------------------------------------------------------------------------------------------------------
        android_radar_calls as (
            SELECT
                concat('a', userid) as userid,
                concat('{date} ', cast(date_add('millisecond', cast(timestamp as bigint), TIME '00:00:00.000') as varchar)) as ts,
                latitude,
                longitude
            FROM app_events.acma
            WHERE date = '{date}'
                and eventname = 'radar_call'
                and service_area_id = '{service_area_id}'
                and date_add('millisecond', cast(timestamp as bigint), TIME '00:00:00.000') between TIME '{start_time}' and TIME '{end_time}'
        )
        ----------------------------------------------------------------------------------------------------------------------------
        select *
        from ios_radar_calls as a
        union all
        select *
        from android_radar_calls as b
    """


def get_wusool_drivers(date: str, service_area_id: int) -> str:
    """
    This query suppose to filter out all non elligible captains, but I found no ways of matching 
    driver_id from prod_stg.driver_ping_parquet with captain_id in prod_dwh.booking.
    This error is not important for Bid Ask Matching research, since all algorithms were tested under different
    supply/demant ratios, but it might be crucial if used somewhere else.
    """
    return f"""
        select
            distinct(cast(captain_id as varchar)) as driver_id
        from prod_dwh.booking
        where day = date('{date}')
            and service_area_id = {service_area_id}
            and captain_id is not null
            and split(cct_name, ' - ')[1] in ('Wusool', 'CareemFood', 'CareemFood RUH.', 'B2B Food', 'English Corporate')
    """
