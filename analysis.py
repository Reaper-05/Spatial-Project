import warnings

import libpysal as lp
import numpy as np
import pandas as pd
import pysal as ps
from sklearn.cluster import DBSCAN


def accident_region(acc_LGA, LGA):
    # Accident in each region
    # Merge them to LGA
    # Count the number of elderly accidents per year
    acc_old_region = \
        acc_LGA[acc_LGA['OLD_DRIVER'] > 0].groupby(['LGA_NAME17'])[
            'ACCIDENT_N'].count()

    LGA_old_each = LGA.merge(acc_old_region, on='LGA_NAME17')

    acc_old = acc_LGA[acc_LGA['OLD_DRIVER'] > 0].groupby(['year'])[
        'ACCIDENT_N'].count()

    return acc_old, LGA_old_each


def add_long_lat(acc_LGA):
    warnings.filterwarnings('ignore')
    acc_old_analysis = acc_LGA[acc_LGA['OLD_DRIVER'] > 0]
    acc_old_analysis['lon'] = acc_old_analysis['geometry'].x
    acc_old_analysis['lat'] = acc_old_analysis['geometry'].y
    warnings.filterwarnings('default')
    return acc_old_analysis


def calc_LGA_old_each(acc_LGA, LGA):
    # Accident in each region
    # Merge them to LGA
    acc_old_region = \
        acc_LGA[acc_LGA['OLD_DRIVER'] > 0].groupby(['LGA_NAME17'])[
            'ACCIDENT_N'].count()

    LGA_old_each = LGA.merge(acc_old_region, on='LGA_NAME17')
    return LGA_old_each


def append_weight_lga_old(LGA_old_each):
    # Create spatial lag and append a column for lag to original geodataframe
    # Prepare standardised value to compare clusterness
    # Create Z values (Standardised value)
    w = ps.lib.weights.Queen.from_dataframe(LGA_old_each,
                                            idVariable='LGA_NAME17')

    LGA_old_each['w_old_acc_LGA'] = lp.weights.lag_spatial(w, LGA_old_each[
        'ACCIDENT_N'])

    LGA_old_each['old_acc_LGA_std'] =\
        (LGA_old_each['ACCIDENT_N'] - LGA_old_each['ACCIDENT_N'].mean())\
        / LGA_old_each['ACCIDENT_N'].std()

    LGA_old_each['w_old_acc_LGA_std'] = lp.weights.lag_spatial(w, LGA_old_each[
        'old_acc_LGA_std'])
    return LGA_old_each, w


def find_clusters(acc_old_analysis_LGA):
    # define the number of kilometers in one radian
    # Coordinates are extracted for later analysis
    # Distance of clusters
    # Haversine method is applied as it works with lon / lat distance
    # Remove the empty list at the tail
    kms_per_radian = 6371.0088

    coords = acc_old_analysis_LGA[['lat', 'lon']].to_numpy()

    epsilon = 0.5 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree',
                metric='haversine').fit(np.radians(coords))

    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series(
        [coords[cluster_labels == n] for n in range(num_clusters)])

    clusters = clusters[0:-1]
    return num_clusters, clusters
