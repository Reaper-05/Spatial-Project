#!/usr/bin/python3

"""This script is boilerplate designed to write test
    cases for assessment 4 of the class
    GEOM90042: Spatial Information Programming.

    Execute by entering your Anaconda environment and
    typing from the command line:

    python test_assessment4.py -v

    The examples here are given from assessment 2, you are
    to create your own and replace the docstrings. Remember
    to keep it simple. The purpose of a unit test is to test
    each operation independently of all others in the program
    to help identify small mistakes.

    Please refer to https://docs.python.org/3/library/unittest.html


    Modified by Darren Christopher Pinto
    Assessment 4 submission.
"""

import unittest
from pathlib import Path
import pandas as pd
import geopandas as gpd
import assessment4 as a4


class TestTaskOne(unittest.TestCase):
    def setUp(self):
        self.filename = Path('data/VicRoadsAccidents/2013/2013.shp')
        self.lga_info = Path('data/RegionsLGA_2017/LGA_2017_VIC.shp')

    """check for reading of shape files"""

    def test_data_frame_import(self):
        shape_data = a4.load_shape_files()
        self.assertTrue(isinstance(shape_data, (list, dict, tuple)))

    """check the crs of the geodataframe  """

    def test_projection_bounds(self):
        data_2013 = gpd.read_file(self.filename)
        lga_data = gpd.read_file(self.lga_info)
        data_2013 = a4.reproject(data_2013, lga_data.crs)
        self.assertTrue(data_2013.crs, lga_data.crs)


class TestTaskTwo(unittest.TestCase):

    def setUp(self):
        self.filename = Path('data/VicRoadsAccidents/2013/2013.shp')
        self.lga_info = Path('data/RegionsLGA_2017/LGA_2017_VIC.shp')
        self.geopackage_name = 'assessment4.gpkg'

    """Check for data present in geopackage  """

    def test_projection(self):
        acc_locations = gpd.read_file(self.geopackage_name,
                                      layer='AccidentLocations')
        self.assertTrue(isinstance(acc_locations, pd.DataFrame))

    """check if division of accident location"""

    def test_severity_byweekend(self):
        acc_locations = gpd.read_file(self.geopackage_name,
                                      layer='AccidentLocations')
        result1, result2 = a4.divide_accidents(acc_locations)
        self.assertTrue(isinstance(result1, pd.DataFrame))
        self.assertTrue(isinstance(result2, pd.DataFrame))


if __name__ == '__main__':
    unittest.main()
