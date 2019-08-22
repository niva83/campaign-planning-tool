import numpy as np
import pandas as pd
from pathlib import Path
import os, shutil
from osgeo import gdal, osr, ogr, gdal_array
import yaml
from xml.dom.minidom import parseString
import dicttoxml
import simplekml
import matplotlib.pyplot as plt
from PIL import Image
import geopandas
from shapely.geometry import Point
import whitebox
import srtm
from pyproj import Proj
from itertools import combinations, product
from random import shuffle