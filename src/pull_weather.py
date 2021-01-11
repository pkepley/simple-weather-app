'''This module contains functions for pulling and saving weather
forecast and actual weather data.'''
import sys
import os
import pathlib
import time
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pytz import timezone
from tzlocal import get_localzone


def random_sleep(sleep_min=3, sleep_max=4):
    '''Sleep for a random amount of time, randomly selected over
    [sleep_min, sleep_max]. Defaults to [3,4].'''
    sleep_time = sleep_min + (sleep_max - sleep_min) * np.random.rand()
    print('Sleeping for {0:04f} seconds.'.format(sleep_time))
    time.sleep(sleep_time)


def repeat_request(url, n_retries=5, sleep_min=15, sleep_max=20):
    '''Repeat a request for data from url up to n_retries times. After
    each failed request, sleep for a random amount of time, randomly
    selected over [sleep_min, sleep_max]. Defaults to [15, 20].'''

    for i in range(n_retries):
        print(f'Attempting to pull from {url}. Attempt {i+1}/{n_retries}')
        try:
            req = requests.get(url)
        except requests.exceptions.RequestException:
            random_sleep(sleep_min, sleep_max)
        else:
            print(f'Requested from {url} with result {req.status_code}')
            if req.status_code == 200:
                print('Successful return. No more retries.')
                break
            else:
                print('Unsuccessful return. Attempting again.')
                random_sleep(sleep_min, sleep_max)
    else:
        req = None

    return req


def pull_nws_fcst(airport_name, df_airports):
    '''Pull the forecast from NWS for the airport specified by
    airport_name, using the information stored on df_airports.'''

    # Get url from table of URLS
    airport_df_row = df_airports[
         df_airports['icao_designation'] == airport_name
    ]

    # Download weather from NWS for given airport
    if len(airport_df_row) == 1:
        url = airport_df_row['nws_fcst_url'].iloc[0]
        req = repeat_request(url)
    else:
        req = None

    # Return result
    return None if req is None else req.text


def parse_nws_fcst(req_text, pull_date_str):
    '''Parse the XML (req_text) containing the forecast pulled from NWS.
    Result will be time-stamped using the pull_date_str.'''

    # Parse result
    tree = ET.ElementTree(ET.fromstring(req_text))
    root = tree.getroot()

    # Data: time and parameters forecasted
    data = root.find('data')
    params = data.find('parameters')

    # Parse the time of forecast
    fcst_time_stamp = [pd.to_datetime(ts.text) for ts in
                       data.find('time-layout').findall('start-valid-time')]

    # Parse the parameters of forecast
    all_series = dict()
    all_series['forecast_time_stamps'] = fcst_time_stamp

    for elem in params:
        if 'type' in elem.attrib:
            series_type = (elem.tag + ' ' + elem.attrib['type'])
            series_type = series_type.replace(' ', '_').replace('-', '_')
            series = [float(v.text) if v.text is not None else np.nan
                      for v in elem.findall('value')]
            all_series[series_type] = series

    # convert to pandas dataframe and write to file
    df = pd.DataFrame(all_series)

    # Append pull date on CSV
    cols = df.columns.to_list()
    df['pull_date'] = pull_date_str
    df = df[['pull_date'] + cols]

    return df


def pull_save_parse_nws_fcst(airport_name, df_airports, pull_date_str,
                             out_dir=None):
    '''Pull, save, and parse the weather forecast from NWS for the airport
    specified by airport_name, using the information stored on
    df_airports. Data will be time-stamped using the date provided on
    pull_date_str.

    If out_dir is provided, then the XML data pulled from NWS will be
    retained there, as will a CSV containing the parsed data.'''

    # Unparsed XML
    req_text = pull_nws_fcst(airport_name, df_airports)

    # Early exit on failure
    if req_text is None:
        return None

    # Retain XML file
    fcst_file_xml = os.path.join(out_dir, f'nws_fcst_{pull_date_str}.xml')
    with open(fcst_file_xml, 'w') as f:
        f.write(req_text)

    # Parse
    df = parse_nws_fcst(req_text, pull_date_str)
    if df is None:
        return df

    # Write CSV to file
    if out_dir is not None:
        fcst_file_name = os.path.join(out_dir, f'nws_fcst_{pull_date_str}.csv')
        print(f'saving to {fcst_file_name}')
        df.to_csv(fcst_file_name, index=False)

    return df


def pull_nws_actl(airport_name, df_airports):
    '''Pull actual weather data from NWS for the airport specified by
    airport_name, using the information stored on df_airports.'''

    # Get url from table of URLS
    airport_df_row = df_airports[
         df_airports['icao_designation'] == airport_name
    ]
    tz_str = airport_df_row['time_zone'].iloc[0]

    # What time is it now?
    now_datetime = datetime.now(get_localzone())
    today_datetime = now_datetime.astimezone(timezone(tz_str))

    # pull data
    base_url = f'https://w1.weather.gov/data/obhistory/{airport_name}.html'
    req = repeat_request(base_url)

    if req is not None:
        req_text = req.text
        req_text = f"<!-- Pulled on : {today_datetime} -->\n" + req_text
    else:
        req_text = None

    return req_text


def parse_nws_actl_raw(req_text):
    '''Parse the full raw HTML (req_text) containing the forecast pulled from NWS.

    No data will be filtered (so the output df will contain 72 hours
    of data), and no date-stamp is added.'''

    # Parse the HTML
    soup = BeautifulSoup(req_text, 'html.parser')

    # extract table data
    table = soup.find_all('table')[3]
    table_head = table.find_all('th')
    table_rows = table.find_all('tr')[3:][:-3]
    table_data = [tr.find_all('td') for tr in table_rows]
    table_data = [[td.text for td in tr] for tr in table_data]
    table_data = np.array(table_data)

    # Get time information from header
    time_cols = [th.text for th in table_head if 'Time' in th]
    if len(time_cols) > 0:
        parse_time_attr = time_cols[0]
        if '(' in parse_time_attr and ')' in parse_time_attr:
            parse_time_zone = parse_time_attr.split('(')[1]
            parse_time_zone = parse_time_zone.split(')')[0]
        else:
            parse_time_zone = None
    else:
        parse_time_zone = None

    # convert table data to dataframe
    df_nws_actl = pd.DataFrame(
         table_data,
         columns=['date', 'time', 'wind', 'visibility', 'weather',
                  'sky_conditions', 'air_temp', 'dew_point',
                  'temp_6_hour_max', 'temp_6_hour_min',
                  'relative_humidity', 'wind_chill', 'heat_index',
                  'pressure', 'pressure_mb', 'precip_1_hour',
                  'precip_3_hour', 'precip_6_hour']
    )

    # convert any NA values to np.nan
    for c in df_nws_actl.columns:
        df_nws_actl[c] = df_nws_actl[c].map(lambda x: '' if x == 'NA' else x)

    # split wind information and rename raw wind data to wind_raw the
    # first line of the following is a lazy hack. it is undone on the
    # last line of this block
    df_nws_actl['wind'] = df_nws_actl['wind'].map(
         lambda x: x + ' 0' if x.lower() in ['calm'] else x
    )
    df_nws_actl['wind_dir'] = df_nws_actl['wind'].map(
         lambda x: x.split(' ')[0]
    )
    df_nws_actl['wind_speed'] = df_nws_actl['wind'].map(
         lambda x:  np.nan if len(x) == 0 else np.float(x.split(' ')[1])
    )
    gust_obs = df_nws_actl['wind'][df_nws_actl['wind'].str.contains('G')]
    gust_speed = gust_obs.map(lambda x: int(x.split(' ')[3]))
    df_nws_actl['gust_speed'] = gust_speed
    df_nws_actl.rename(columns={'wind': 'wind_raw'}, inplace=True)
    df_nws_actl['wind_raw'] = df_nws_actl['wind_raw'].map(
         lambda x: 'Calm' if x.lower() == 'calm 0' else x
    )

    # remove percent signs from humidity
    df_nws_actl['relative_humidity'] = df_nws_actl['relative_humidity'].map(
         lambda x: x.replace('%', '')
    )

    # the following variables are numeric
    numeric_var_list = ['visibility', 'air_temp', 'dew_point',
                        'relative_humidity', 'temp_6_hour_max',
                        'temp_6_hour_min', 'pressure', 'pressure_mb',
                        'precip_1_hour', 'precip_3_hour', 'precip_6_hour',
                        'wind_chill', 'heat_index']

    # convert the numeric variables to floats
    for v in numeric_var_list:
        tmp = df_nws_actl[v][np.where(df_nws_actl[v] != '')[0]]
        tmp = tmp.astype(np.float)
        df_nws_actl[v] = tmp

    # reorder columns back to original order
    df_nws_actl = df_nws_actl[[
        'date', 'time', 'wind_raw', 'wind_dir', 'wind_speed',
        'gust_speed', 'visibility', 'weather', 'sky_conditions',
        'air_temp', 'dew_point', 'temp_6_hour_max', 'temp_6_hour_min',
        'relative_humidity', 'wind_chill', 'heat_index', 'pressure',
        'pressure_mb', 'precip_1_hour', 'precip_3_hour', 'precip_6_hour'
    ]]

    return df_nws_actl, parse_time_zone


# Dictionaries are defined here to deal with the fact that times are
# converted on the webpage prior to the true DST conversion. So times
# are harder to parse on these dates
dst_timezone_alts = {'US/Eastern': 'Etc/GMT+4',
                     "US/Central": 'Etc/GMT+5',
                     "US/Mountain": 'Etc/GMT+6',
                     "US/Pacific": 'Etc/GMT+7',
                     "US/Alaska": 'Etc/GMT+8',
                     "US/Hawaii": 'US/Hawaii'}


non_dst_timezone_alts = {'US/Eastern': 'Etc/GMT+5',
                         "US/Central": 'Etc/GMT+6',
                         "US/Mountain": 'Etc/GMT+7',
                         "US/Pacific": 'Etc/GMT+8',
                         "US/Alaska": 'Etc/GMT+9',
                         "US/Hawaii": 'US/Hawaii'}


def parse_nws_actl(req_text, df_airports, airport_name, date_str_last_actl,
                   date_str_retain=None):
    '''Parse the raw HTML (req_text) containing the forecast pulled from NWS.

    Only the date indicated by date_str_retain will be kept in the
    output dataframe. Also, times will be corrected for the
    time-zone.'''

    # Get url from table of URLS
    airport_df_row = df_airports[
         df_airports['icao_designation'] == airport_name
    ]
    tz_str = airport_df_row['time_zone'].iloc[0]

    # Datetime represents midnight on the last actual date for the pull
    datetime_last_actl = timezone(tz_str).localize(
        datetime.strptime(date_str_last_actl, '%Y-%m-%d')
    )

    # Parse the raw HTML
    df_nws_actl, parse_time_zone = parse_nws_actl_raw(req_text)

    # Is it daylight savings time?
    is_dst = 'DT' in parse_time_zone.upper()

    # Early exit if None
    if df_nws_actl is None:
        return None

    # Get columns
    cols = df_nws_actl.columns.to_list()

    # Last date does not match so parsing will fail. Return None to
    # avoid errors.
    if datetime_last_actl.strftime('%d') not in df_nws_actl['date'].to_list():
        print('Expected LA date was: {}.'.format(
            datetime_last_actl.strftime('%d'))
        )
        print('Dataset  LA date was: {}.'.format(
            df_nws_actl.iloc[0]['date'])
        )
        return None

    # What is the D/M/Y for the LA date?
    year = datetime_last_actl.year
    month = datetime_last_actl.month
    day = datetime_last_actl.day

    # Add day column for datetime computation
    df_nws_actl['day'] = df_nws_actl['date'].astype(str).astype(int)

    # Add month column for datetime computation. If date increases in past,
    # this signifies we've crossed a month. Adjust accordingly.
    df_nws_actl['month'] = month
    df_nws_actl['month'] = df_nws_actl['month'] - (df_nws_actl['day'] > day)

    # Add year column for datetime computation. If month is January
    # and date increases in past, this signifies we've crossed a year.
    df_nws_actl['year'] = year
    if month == 1:
        df_nws_actl['year'] = df_nws_actl['year'] - (df_nws_actl['day'] > day)

    # Add hour and minute column for datetime computation
    df_nws_actl['hour'] = df_nws_actl['time'].apply(
        lambda x: int(x.split(':')[0])
    )
    df_nws_actl['minute'] = df_nws_actl['time'].apply(
        lambda x: int(x.split(':')[1])
    )

    # Naive Datetime
    df_nws_actl['datetime'] = pd.to_datetime(
         df_nws_actl[['year', 'month', 'day', 'hour', 'minute']]
    )

    # Be careful about transitions to DST!
    if is_dst:
        tz_str_alt = dst_timezone_alts[tz_str]
        df_nws_actl['datetime'] = df_nws_actl['datetime'].apply(
            lambda x: timezone(tz_str_alt).localize(x)
        )
        df_nws_actl['datetime'] = df_nws_actl['datetime'].apply(
            lambda x: x.astimezone(tz_str)
        )
    else:
        tz_str_alt = non_dst_timezone_alts[tz_str]
        df_nws_actl['datetime'] = df_nws_actl['datetime'].apply(
            lambda x: timezone(tz_str_alt).localize(x)
        )
        df_nws_actl['datetime'] = df_nws_actl['datetime'].apply(
            lambda x: x.astimezone(tz_str)
        )

    # ensure matching
    df_nws_actl['date'] = df_nws_actl['datetime'].apply(
        lambda x: x.strftime('%d')
    )
    df_nws_actl['time'] = df_nws_actl['datetime'].apply(
        lambda x: x.strftime('%H:%M')
    )

    # drop the temp columns
    df_nws_actl = df_nws_actl.drop(
        columns=['year', 'month', 'day', 'hour', 'minute']
    )

    # Re-order
    df_nws_actl = df_nws_actl[['datetime'] + cols]

    # only retain requested day
    if date_str_retain is not None:
        datetime_retain = timezone(tz_str).localize(
            datetime.strptime(date_str_retain, '%Y-%m-%d')
        )
        date_retain = datetime_retain.strftime('%d')
        df_nws_actl = df_nws_actl[df_nws_actl.date == date_retain]

    return df_nws_actl


def pull_save_parse_nws_actl(airport_name, df_airports, out_dir=None):
    '''Pull, save, and parse the actual weather data from NWS for the
    airport specified by airport_name, using the information stored on
    df_airports. Data will be time-stamped using the date provided on
    pull_date_str.

    If out_dir is provided, then the HTML data pulled from NWS will be
    retained there, as will a CSV containing the parsed data.'''

    # Get url from table of URLS
    airport_df_row = df_airports[
        df_airports['icao_designation'] == airport_name
    ]
    tz_str = airport_df_row['time_zone'].iloc[0]

    # Current time
    datetime_last_actl = datetime.now(get_localzone())
    datetime_last_actl = datetime_last_actl.astimezone(timezone(tz_str))
    if datetime_last_actl.hour < 12:
        datetime_last_actl = datetime_last_actl + timedelta(hours=-12)
    date_str_last_actl = datetime_last_actl.strftime('%Y-%m-%d')

    # Get the request
    req_text = pull_nws_actl(airport_name, df_airports)

    if req_text is None:
        return None

    # Retain html file
    actl_file_html = os.path.join(
        out_dir, f'nws_actl_{date_str_last_actl}.html'
    )
    with open(actl_file_html, 'w') as f:
        f.write(req_text)

    # Parse and create dataframe
    df_nws_actl = parse_nws_actl(req_text, df_airports, airport_name,
                                 date_str_last_actl, date_str_last_actl)

    # If parsing failed, early return
    if df_nws_actl is None:
        return None

    # Write parsed CSV to file
    actl_file_name = os.path.join(out_dir,
                                  f'nws_actl_{date_str_last_actl}.csv')
    print(f'saving to {actl_file_name}')
    df_nws_actl.to_csv(actl_file_name, index=False)

    return df_nws_actl


def pull_and_save(df_airports, df_airports_to_pull, pull_date_str,
                  out_root=None):
    '''Pull and save the weather forecast and actual weather data for the
    airports indicated in df_airports_to_pull, using the information
    stored in df_airports.

    Data will be time stamped with the timestamp stored in
    pull_date_str.

    If out_root is provided, results will be saved there.

    '''

    # Iterate through list of airports and pull data
    for _, row in df_airports_to_pull.iterrows():
        airport_name = row['icao_designation']

        # Create output dir for airport if it doesn't exist
        if out_root is not None:
            airport_out_dir = os.path.join(out_root, airport_name)
            pathlib.Path(airport_out_dir).mkdir(parents=True, exist_ok=True)
        else:
            airport_out_dir = None

        # Pull Forecast
        try:
            print(f'Attempting to pull forecast for {airport_name}.')
            df_fcst = pull_save_parse_nws_fcst(airport_name,
                                               df_airports, pull_date_str,
                                               out_dir=airport_out_dir)
            if df_fcst is not None:
                print(f'Successful fcst pull for {airport_name}' +
                      f'on {pull_date_str}.')
            else:
                print(f'No fcst result was returned for {airport_name}' +
                      f'for {pull_date_str}.')
        except ValueError:
            print(f'Error while pulling fcst for {airport_name}.')

        # Pull actual data
        try:
            print(f'Attempting to pull actual for {airport_name}.')
            df_actl = pull_save_parse_nws_actl(airport_name, df_airports,
                                               out_dir=airport_out_dir)
            if df_actl is not None:
                print(f'Successful actl pull for {airport_name}' +
                      f'on {pull_date_str}.')
            else:
                print(f'No actl result was returned for {airport_name}' +
                      f'for {pull_date_str}.')
        except ValueError:
            print(f'Error while pulling actl for {airport_name}.')

    # Sleep for a few seconds
    random_sleep()
    sys.stdout.flush()


def midnight_time_zone(pull_hour=0):
    '''Get the US timezone whose current time is pull_hour. pull_hour
    defaults to 0, so that we will select the timezone which is
    currently midnight by default.'''

    # What time is it now in our current timezone?
    local_now_datetime = datetime.now(get_localzone())

    # Set up US timezones
    us_tz_strs = ["US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
                  "US/Alaska", "US/Hawaii"]

    # Timezone to run (ie where it's midnight)
    tz_to_run = None

    # Print local zone:
    now_datetime_in_tz = [local_now_datetime.astimezone(timezone(tz_str)) for
                          tz_str in us_tz_strs]
    now_hour_in_tz = [dt.hour for dt in now_datetime_in_tz]

    # In which timezone is it midnight?
    midnight_hour = pull_hour
    if midnight_hour in now_hour_in_tz:
        idx_tz_to_run = now_hour_in_tz.index(midnight_hour)
        tz_to_run = us_tz_strs[idx_tz_to_run]
        pull_datetime = now_datetime_in_tz[idx_tz_to_run]
        pull_date_str = pull_datetime.strftime('%Y-%m-%d')
        print(f'Will update db for airports with time-zone {tz_to_run}')
    else:
        pull_date_str = None

    return tz_to_run, pull_date_str


def midnight_pull_df(df_airports, pull_hour=0):
    '''Pull a dataframe containing the nws actual/forecast weather data
    for all airports specified on df_airports whose current time is
    pull_hour in their respective time-zone. pull_hour defaults to 0,
    so that we will pull data only at midnight by default.'''

    # what time zone is it midnight in?
    tz_to_run, pull_date_str = midnight_time_zone(pull_hour=pull_hour)

    # List of airports to pull
    df_airports_to_pull = df_airports[df_airports['time_zone'] == tz_to_run]

    return df_airports_to_pull, pull_date_str


def midnight_pull_and_save(df_airports, out_root=None, pull_hour=0):
    '''Pull and save the nws actual/forecast weather data for all airports
    specified on df_airports whose current time is pull_hour in their
    respective time-zone. pull_hour defaults to 0, so that we will
    pull data only at midnight by default.'''

    # Grab the list if airports for this range
    df_airports_to_pull, pull_date_str = midnight_pull_df(
        df_airports, pull_hour=pull_hour
    )
    pull_and_save(df_airports, df_airports_to_pull, pull_date_str, out_root)
