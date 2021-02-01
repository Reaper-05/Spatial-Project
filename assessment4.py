"""
Supporting Module for assessment 4.
Authors:    Darren Christopher Pinto StudentID: 1033936
            Email: darrenchrist@student.unimelb.edu.au

            Hirokazu Saigusa StudentID: 9722011
            Email: hsaigusa@student.unimelb.edu.au
Module with file read and  file manipulations, extractions
and writing.


"""
import warnings
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import shape

ACC_2013_FILE = 'data/VicRoadsAccidents/2013/2013.shp'
ACC_2014_FILE = 'data/VicRoadsAccidents/2014/2014.shp'
ACC_2015_FILE = 'data/VicRoadsAccidents/2015/2015.shp'
ACC_2016_FILE = 'data/VicRoadsAccidents/2016/2016.shp'
ACC_2017_FILE = 'data/VicRoadsAccidents/2017/2017.shp'
ACC_2018_FILE = 'data/VicRoadsAccidents/2018/2018.shp'
LAYOUT_FILE_LGA = 'data/RegionsLGA_2017/LGA_2017_VIC.shp'
LAYOUT_FILE_SA2 = 'data/RegionsSA2_2016/SA2_2016_AUST.shp'
ACCIDENT_DATE = 'ACCIDENT_D'
ACCIDENT_NO = 'ACCIDENT_N'
YEAR = 'year'
ACCIDENT_1 = 'ACCIDENT_1'
COUNT = 'count'
ACC_PCT = 'acc_pct'
NO_OF_VEHICLE = 'NO_OF_VEHI'
HEAVY_VEHICLE = 'HEAVYVEHIC'
PASSENGER_VEHICLE = 'PASSENGERV'
MOTOR_CYCLE = 'MOTORCYCLE'
PUBLIC_VEHICLE = 'PUBLICVEHI'
POLYGON = 'Polygon'
MULTIPOLYGON = 'MultiPolygon'
YEAR_2013 = '2013'
POINT = 'Point'
YEAR_2014 = '2014'
YEAR_2015 = '2015'
YEAR_2016 = '2016'
YEAR_2017 = '2017'
YEAR_2018 = '2018'
NO_ = 'No.'
DIFF_ = 'Diff.'
CHANGE = 'Change'
LGA_NAME17 = 'LGA_NAME17'
DAY_OF_WEE = 'DAY_OF_WEE'
TOTAL_PERS = 'TOTAL_PERS'
SEVERITY = 'SEVERITY'


def load_shape_files():
    """
    Load the shape files of disk and append to the list
    :return: a list of shape containing accident data.
    """
    accident_list = []
    try:
        if Path(ACC_2013_FILE).is_file():
            accident_list.append(gpd.read_file(ACC_2013_FILE))
        if Path(ACC_2014_FILE).is_file():
            accident_list.append(gpd.read_file(ACC_2014_FILE))
        if Path(ACC_2015_FILE).is_file():
            accident_list.append(gpd.read_file(ACC_2015_FILE))
        if Path(ACC_2016_FILE).is_file():
            accident_list.append(gpd.read_file(ACC_2016_FILE))
        if Path(ACC_2017_FILE).is_file():
            accident_list.append(gpd.read_file(ACC_2017_FILE))
        if Path(ACC_2018_FILE).is_file():
            accident_list.append(gpd.read_file(ACC_2018_FILE))
    except IOError:
        print('Error in opening file')
    return accident_list


def read_LGA():
    """
    Load the shape file of disk and append to the list
    :return: a list of shape containing LGA data.
    """
    try:
        if Path(LAYOUT_FILE_LGA).is_file():
            lga_df = gpd.read_file(LAYOUT_FILE_LGA)
    except IOError:
        print('Error in opening file')
    return lga_df


def read_SA2():
    """
    Load the shape file of disk and append to the list
    :return: a list of shape containing SA2 region data.
    """
    try:
        if Path(LAYOUT_FILE_SA2).is_file():
            sa2_df = gpd.read_file(LAYOUT_FILE_SA2)
    except IOError:
        print('Error in opening file')
    return sa2_df


def merge_shape_files(accident_list):
    """
    # Concatonate accident data of all years
    :param accident_list: data frames of all accidents Between years 2013-2018
    :return:
    """
    return pd.concat(accident_list)


def add_year_column(merged_data):
    """
    # Extract years for later analysis
    Since shape data is not accurate with date, we
    extract the info manually and assign the years.

    :param merged_data: data from all year shape files
    :return: a data-frame containing new field called year
    """

    year_list = [x.strip()[-4:] for x in merged_data[ACCIDENT_DATE]]
    return merged_data.assign(year=year_list)


def extract_avg_accidents(total_data):
    """
    Get the avg no. of accidents per year
    :param total_data: total merged data with year
    :return: a dataframe with avg accidents
    """

    try:
        avg_per_year = (total_data[ACCIDENT_NO].count() / total_data[
            YEAR].nunique()).round().astype(np.int64)
    except KeyError:
        print(' Given key not found,please check the parameters')
    return avg_per_year


def calc_accident_percent(total_data):
    """
    Get the calc accident percent per year
    :param total_data: total merged data with year
    :return: A tuple with 2 highest percent of accidents
    """

    acc_total = total_data.groupby(ACCIDENT_1)[
        ACCIDENT_NO].count().sort_values(
        ascending=False).to_frame().reset_index().rename({0: COUNT}, axis=1)
    acc_total[ACC_PCT] = ((acc_total.ACCIDENT_N / acc_total.ACCIDENT_N.sum())
                          * 100).round(1)

    acc_1 = (acc_total.loc[0, ACCIDENT_NO], acc_total.loc[0, ACC_PCT])
    acc_2 = (acc_total.loc[1, ACCIDENT_NO], acc_total.loc[1, ACC_PCT])

    return (acc_1, acc_2)


def find_vehicle_distn(total_data):
    """
    find the vehicles involved in accident by distribution
    :param total_data: total merged data with year
    :return:
    """
    return total_data.groupby(YEAR)[
        [NO_OF_VEHICLE, HEAVY_VEHICLE, PASSENGER_VEHICLE, MOTOR_CYCLE,
         PUBLIC_VEHICLE]].sum()


def clean_lga_df(lga_df):
    """
    Clean the the lga-data frame and remove all none(null types in geometry)
    :param lga_df:  lga geo_dataframe
    :return: cleaned geo data frame
    """
    return lga_df[(lga_df.geometry.type == POLYGON) | (
            lga_df.geometry.type == MULTIPOLYGON)]


def clean_total_data(total_df):
    """
    Clean the the total_df frame and remove all none(null types in geometry)
    :param total_df:  total geo_dataframe
    :return: cleaned total data frame
    """

    return total_df[total_df.geometry.type == POINT]


def reproject(data_frame, crs):
    """
     re-project the data frame to given crs
     :param data_frame: the input geo-data frame
     :param crs: the resulting crs
     :return: the reprojected data-frame
     """
    try:
        warnings.filterwarnings('ignore')
        data_frame.to_crs(crs, inplace=True)
        warnings.filterwarnings('default')
    except ValueError:
        print('None types present in Geometry')
    return data_frame


def add_headers(result_ordered, acc13_keys):
    """
    Add headers for the result to be shown in particular style
    :param result_ordered: get the ordered data
    :param acc13_keys: the keys( lga names)
    :return: a data-frame with headers and new keys for indexing
    """

    s1 = pd.Series(
        [YEAR_2013, YEAR_2014, YEAR_2014, YEAR_2014, YEAR_2015, YEAR_2015,
         YEAR_2015, YEAR_2016, YEAR_2016, YEAR_2016, YEAR_2017, YEAR_2017,
         YEAR_2017, YEAR_2018, YEAR_2018, YEAR_2018])

    s2 = pd.Series(
        [NO_, NO_, DIFF_, CHANGE, NO_, DIFF_, CHANGE, NO_, DIFF_, CHANGE, NO_,
         DIFF_, CHANGE, NO_, DIFF_, CHANGE])

    return result_ordered.T.set_index([s1, s2]).T.reindex(acc13_keys)


def generate_gdf(vehicle_distn, total_data):
    """
    Create a geodataframe for storing as layer in geopackage
    :param vehicle_distn: a dataframe with  vehicle data
    :param total_data:  total geo_dataframe
    :return:  vehicle geo_dataframe
    """
    geo_vehicle_distn = total_data
    return gpd.GeoDataFrame(
        vehicle_distn, crs=total_data.crs,
        geometry=geo_vehicle_distn.dissolve(by='year',
                                            aggfunc='mean').geometry)


def get_acc_lga(total_data, lga_df):
    """
    Spatial join between lga and total_data.
    :param total_data: total geo_dataframe
    :param lga_df: lga geodataframe
    :return: a total geo_dataframe with lga boundaries
    """
    return gpd.sjoin(total_data, lga_df, how='left', op='intersects')


def calc_accidents_bylga(total_data, lga_df):
    """
    Extract keys ordered in descending manner for the final table order
    get Top 10 count
    get Top10 percentage change
    :param total_data: total geo_dataframe
    :param lga_df: lga dataframe
    :return: top 10 lga with highest accident count
    """
    try:
        acc_lga = get_acc_lga(total_data, lga_df)

        acc_2013_t10 = acc_lga[acc_lga[YEAR] == YEAR_2013].groupby(LGA_NAME17)[
            ACCIDENT_NO].count().sort_values(
            ascending=False).head(10)

        acc13_keys = acc_2013_t10.keys()
        acc_rank = \
            acc_lga.loc[acc_lga[LGA_NAME17].isin(acc_2013_t10.index)].groupby(
                [YEAR, LGA_NAME17])[ACCIDENT_NO].count()

        acc_top_tran = acc_rank.unstack(level=0)
        acc_top_diff = acc_top_tran.diff(axis=1)

        acc_top_chan = acc_top_tran.pct_change(axis=1).round(2)
        result = pd.concat([acc_top_tran, acc_top_diff, acc_top_chan], axis=1)
        result_ordered = result[
            [item for items in zip(acc_top_tran.columns) for item in
             items]].dropna(axis=1)
    except KeyError:
        print(' Given key not found,please check the parameters')

    return add_headers(result_ordered, acc13_keys)


def calc_accidents_byweek(total_data):
    """
    Extract values for error bars
    Make the right order
    :param total_data: total geo_dataframe
    :return: a data frame with accident by weeks
    """

    acc_week = total_data.groupby([YEAR, DAY_OF_WEE])[ACCIDENT_NO].count()
    acc_week = acc_week[[YEAR_2013, YEAR_2018]]
    acc_week_df = acc_week.unstack(level=1)

    acc_week_ordered = acc_week_df[
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
         'Sunday']].T
    return acc_week_ordered


def calc_accidents_byservity(total_data):
    """
    Create a table of accident severity transition 2013 - 2018
    :param total_data:
    :return:
    """
    acc_severity = total_data.groupby([YEAR, SEVERITY])[ACCIDENT_NO].count()
    return acc_severity.unstack(level=1)


def format_string(result):
    """
    Format the result with ','
    :param result: a string
    :return: a string
    """
    if len(result) > 0:
        result += ', '
    return result


def get_vehicle_list(vehicles):
    """
    Append a string for each vehicle type
    :param vehicles: get the list of vehicles
    :return: a string with result
    """
    # heavy_ve, passenger_ve, motorcycle, public_ve
    # note the missing vehicle type. Question: Add bicyclist?Complete missing
    result = ""
    if vehicles[0] > 0:
        result += 'Heavy Vehicle '
    if vehicles[1] > 0:
        result = format_string(result)
        result += 'Motor cycle '
    if vehicles[2] > 0:
        result = format_string(result)
        result += 'Passenger Vehicle '
    if vehicles[3] > 0:
        result = format_string(result)
        result += 'Public Vehicle'
    if result == '':
        result = 'Vehicle Unknown'

    return result


def filter_3people(total_data):
    """
    take the data when 3 or more ppl are involved
    :param total_data: total geo data
    :return: filtered geo data
    """
    return total_data[total_data[TOTAL_PERS] > 2]


def add_accident_locations(total_data):
    """
    Accident location layer data.
    :param total_data: total geo data
    :return: geo dataframe with selected data
    """
    accident_loctns_df = pd.DataFrame()
    try:
        accident_loctns_df['AccidentNumber'] = total_data[ACCIDENT_NO]
        accident_loctns_df['VehicleType'] = total_data[
            [HEAVY_VEHICLE, MOTOR_CYCLE, PASSENGER_VEHICLE,
             PUBLIC_VEHICLE]].apply(get_vehicle_list, axis=1)
        accident_loctns_df['DayOfWeek'] = total_data[DAY_OF_WEE]
        accident_loctns_df['NumPeople'] = total_data[TOTAL_PERS]
        accident_loctns_df['Severity'] = total_data[SEVERITY]
        accident_loctns_gdf = gpd.GeoDataFrame(
            accident_loctns_df, crs=total_data.crs,
            geometry=total_data.geometry)
    except KeyError:
        print(' Given key not found,please check the parameters')

    return accident_loctns_gdf


def spatial_index_join(layer_data_gfd, shp_file):
    """
    this function is used for timeit properyt
    A spatial join with custom column renaming as per the requirements
    This method is mentioned as spatial join-using index by the documentation
    :param layer_data_gfd: layer_data geodata frame
    :param shp_file: shape file_ SA2 regions
    :return: A geodata frame with SA2 info
    """

    try:
        joined_gdf = gpd.sjoin(layer_data_gfd,
                               gpd.GeoDataFrame(shp_file['SA2_NAME16'],
                                                geometry=shp_file.geometry),
                               how='left', op='intersects')
        joined_gdf.rename(columns={'SA2_NAME16': 'SA2'}, inplace=True)
    except KeyError:
        print(' Given key not found,please check the parameters')

    return joined_gdf


def format_fordisplay(vehicle_distn):
    """
    Vehicle distribution table with correct column headers
    :param vehicle_distn: data frame containing vehicle distribution
    :return: data frame with formatted headers
    """
    vehicle_distn.rename({'NO_OF_VEHI': 'No of Vehicles',
                          'HEAVYVEHIC': 'Heavy Vehicles',
                          'PASSENGERV': 'Passenger Vehicles',
                          'MOTORCYCLE': 'Motorcycle',
                          'PUBLICVEHI': 'Public Vehicle'},
                         inplace=True)

    vehicle_distn = vehicle_distn.rename_axis('Year', axis=1)
    return vehicle_distn


def find_sa2(loc, sa2):
    """
    Method for naive checking and adding of  SA2 region by using shapely
    :param loc:
    :param sa2:
    :return:
    """
    point = shape(loc)
    for i, data in sa2.iterrows():
        if point.within(data.geometry):
            return data['SA2_NAME16']
    return None


def add_SA2_naive_method(layer_data_gfd, sa2):
    """
    This method is used only for timeit implementation purpose
    SA2 data is added into layer data
    :param layer_data_gfd: accident locations data frame
    :param sa2: sa2 geo data frame.
    :return: accident locations data frame with sa2 info
    """
    layer_data_gfd['SA2'] = layer_data_gfd.geometry.apply(
        lambda x: find_sa2(x, sa2))
    return layer_data_gfd


def get_accidents_byweekends(acc_locations):
    """
    find accidents on weekends and based on serious or fatal accidents
    :param acc_locations: acc_location geo data frame
    :return: geodataframe with required data
    """
    return acc_locations[((acc_locations.DayOfWeek == 'Saturday') | (
            acc_locations.DayOfWeek == 'Sunday')) &
                         ((acc_locations.Severity == 'Serious injury accident')
                          | (acc_locations.Severity == 'Fatal accident'))]


def get_accidents_byweekdays(acc_locations):
    """
    find accidents on weekdays and based on serious or fatal accidents
    :param acc_locations: acc_location geo data frame
    :return: geodataframe with required data
    """
    return acc_locations[((acc_locations.DayOfWeek != 'Saturday') & (
            acc_locations.DayOfWeek != 'Sunday') & (
                              acc_locations.DayOfWeek.notnull())) &
                         ((acc_locations.Severity == 'Serious injury accident')
                          | (acc_locations.Severity == 'Fatal accident'))]


def divide_accidents(acc_locations):
    """
    Return a tuple with weekdays and weekends data
    :param acc_locations: acc_location geo data frame
    :return: a tuple to geo data frames.
    """
    return get_accidents_byweekends(acc_locations), get_accidents_byweekdays(
        acc_locations)


def get_normalized_data_choro(acc_lga, lga_df):
    """
    get data normalised for choropleth
    :param acc_lga: accident locations
    :param lga_df: lga data frame
    :return: a normalized data frame for dispaly
    """

    try:
        total_accidents_lga = acc_lga.groupby([YEAR, LGA_NAME17])[
            LGA_NAME17].count()
        total_accidents_lga = total_accidents_lga.unstack(level=0)
        total_accidents_lga.reset_index()
        biggest_acc = acc_lga.groupby(SEVERITY)[
            ACCIDENT_NO].count().sort_values(ascending=False).head(1)
        acc_sev = \
            acc_lga[
                acc_lga[SEVERITY] == biggest_acc.index.tolist()[0]].groupby(
                [LGA_NAME17, YEAR])[
                'ACCIDENT_N'].count().sort_values(ascending=False)
        acc_sev = acc_sev.unstack(level=1)

        acc_sev = acc_sev.div(total_accidents_lga).reset_index()
        acc_lga_merged = gpd.GeoDataFrame(
            acc_sev.merge(
                gpd.GeoDataFrame(lga_df[LGA_NAME17], geometry=lga_df.geometry),
                on=LGA_NAME17),
            crs=lga_df.crs)
    except KeyError:
        print(' Given key not found,please check the parameters')

    return acc_lga_merged


def get_data_choro(acc_lga, lga_df):
    """
    Get default data to show rate of change
    :param acc_lga:
    :param lga_df:
    :return:  a data frame to display
    """
    try:
        biggest_acc = acc_lga.groupby(SEVERITY)[
            ACCIDENT_NO].count().sort_values(ascending=False).head(1)

        acc_sev = \
            acc_lga[
                acc_lga[SEVERITY] == biggest_acc.index.tolist()[0]].groupby(
                [LGA_NAME17, YEAR])[
                ACCIDENT_NO].count().sort_values(ascending=False)
        acc_sev = acc_sev.unstack(level=1)

        acc_lga_merged = gpd.GeoDataFrame(
            acc_sev.merge(
                gpd.GeoDataFrame(lga_df[LGA_NAME17], geometry=lga_df.geometry),
                on=LGA_NAME17),
            crs=lga_df.crs)
    except KeyError:
        print(' Given key not found,please check the parameters')

    return acc_lga_merged
