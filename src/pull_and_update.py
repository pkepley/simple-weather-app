import pathlib
from datetime import datetime
from tzlocal import get_localzone

from pandas import read_csv
from db_setup import data_root, airport_list_loc
from pull_weather import midnight_pull_and_save

# Make sure output path exists
pathlib.Path(data_root).mkdir(parents=True, exist_ok=True)

# Get the airport list
df_airports = read_csv(airport_list_loc)

# What time is it now in our current timezone?
local_time_zone = get_localzone()
local_now_datetime = datetime.now(local_time_zone)
print(f'Local now ({local_time_zone}) : {local_now_datetime}')

# Update pull and save
pull_hour = 0
midnight_pull_and_save(df_airports, out_root=data_root,
                       pull_hour=pull_hour)
