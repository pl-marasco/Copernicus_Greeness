import xarray as xr
import pandas as pd
from distributed import Client
import os
import rasterio as rs
import cv2
import numpy as np
from skimage.color import rgb2hsv
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd


def _files(path):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.tiff'):
                if 'B02' in filename:
                    list_of_files['B02'] = os.sep.join([dirpath, filename])
                elif 'B03' in filename:
                    list_of_files['B03'] = os.sep.join([dirpath, filename])
                elif 'B04' in filename:
                    list_of_files['B04'] = os.sep.join([dirpath, filename])
                elif 'B08' in filename:
                    list_of_files['B08'] = os.sep.join([dirpath, filename])
                elif 'B8A' in filename:
                    list_of_files['B8A'] = os.sep.join([dirpath, filename])
                elif 'B11' in filename:
                    list_of_files['B11'] = os.sep.join([dirpath, filename])

    return list_of_files


def _ndvi(red, nir):

    return (nir - red) / (nir + red)


def _hls(mir, nir, red):

    rgb = np.dstack((mir[0]/1e4, nir[0]/1e4, red[0]/1e4))
    return rgb2hsv(rgb)


def _normalizer(dataset):

    data = dataset.flatten()
    return (data-data.mean())/data.std()


def _indexes(B04, B8A, B11):

    NDVI_v = _ndvi(B04.data, B8A.data)
    H_v = _hls(B11.data, B8A.data, B04.data)

    x = B04.x
    y = B04.y

    NDVI = xr.DataArray(NDVI_v[0, :, :], dims=('y', 'x'), coords={'x': x, 'y': y})
    H = xr.DataArray(H_v[:, :, 0], dims=('y', 'x'), coords={'x': x, 'y': y})

    return NDVI, H


def _poi(f_path, date, value):

    table = gpd.read_file(f_path)

    # TODO add a date range selection
    sub_table = table[table.LC == value]
    # cleaned = sub_table.drop('')

    return sub_table


def _selection(NDVI, H):

    NDVI_pos = np.where(NDVI < 0, np.nan, NDVI)
    NDVI_flat = np.where(NDVI > 0.5, np.nan, NDVI_pos).flatten()
    H = H * 360.
    H_flat = np.where(H > 80, np.nan, H).flatten()

    bad = ~np.logical_or(np.isnan(NDVI_flat), np.isnan(H_flat))
    NDVI_cln = np.compress(bad, NDVI_flat)
    H_cln = np.compress(bad, H_flat)

    return NDVI_cln, H_cln


def _point_extraction(da, xy_poi):

    values = []
    for index, row in xy_poi.iterrows():
        values.append(da.sel(x=row.x, y=row.y, method="nearest").data)

    values = np.array(values)
    return values


if __name__ == '__main__':

    path = r'C:\Data\HSV\Observations'

    for (dirpath, dirnames, filenames) in os.walk(path):
        for dirname in dirnames:
            bands_pth = _files(os.sep.join([dirpath, dirname]))

            B04 = xr.open_rasterio(bands_pth['B04'])
            B08 = xr.open_rasterio(bands_pth['B08'])
            B8A = xr.open_rasterio(bands_pth['B8A'])
            B11 = xr.open_rasterio(bands_pth['B11'])

            date = None

            # TODO valido solo su una data
            xy_poi_g = _poi(r'C:\Data\HSV\POI\points.dbf', date, 1)
            xy_poi_b = _poi(r'C:\Data\HSV\POI\points.dbf', date, 0)

            NDVI_g, H_g = _indexes(B04, B8A, B11)
            NDVI_b, H_b = _indexes(B04, B8A, B11)

            NDVI_g = _point_extraction(NDVI_g, xy_poi_g)
            H_g = _point_extraction(H_g, xy_poi_g)

            NDVI_b = _point_extraction(NDVI_b, xy_poi_b)
            H_b = _point_extraction(H_b, xy_poi_b)

            NDVI_rdc_g, H_rdc_g = _selection(NDVI_g, H_g)
            NDVI_rdc_b, H_rdc_b = _selection(NDVI_b, H_b)

            plt.figure(figsize=(20, 10))
            plt.scatter(x=NDVI_rdc_g, y=H_rdc_g, s=5, marker='<', c='green')
            plt.scatter(x=NDVI_rdc_b, y=H_rdc_b, s=5, marker='>', c='red')

            plt.show()

            # size = 25000
            #
            # rand = np.random.choice(NDVI_cln.size, size)
            # MB_matrix = np.zeros((size, 2))
            #
            # MB_matrix[:, 0] = np.take(NDVI_cln, rand)
            # MB_matrix[:, 1] = np.take(H_cln, rand)
            #
            # plt.figure(figsize=(20, 10))
            #
            # MB_matrix = np.zeros((NDVI_cln.size, 2))
            #
            # MB_matrix[:, 0] = NDVI_cln
            # MB_matrix[:, 1] = H_cln
            #
            # plt.scatter(x=MB_matrix[:, 0], y=MB_matrix[:, 1], s=1, marker='2')

            # plt.show()



            # # Covariance
            # np.set_printoptions(precision=3)
            # cov = np.cov(MB_matrix.transpose())
            #
            # # Eigen Values
            # EigVal, EigVec = np.linalg.eig(cov)
            # print("Eigenvalues:\n\n", EigVal, "\n")
            #
            # # Ordering Eigen values and vectors
            # order = EigVal.argsort()[::-1]
            # EigVal = EigVal[order]
            # EigVec = EigVec[:, order]
            #
            # # Projecting data on Eigen vector directions resulting to Principal Components
            # PC = np.matmul(MB_matrix, EigVec)  # cross product
            #
            # # Generate Paiplot for original data and transformed PCs
            #
            # Bandnames = ['Band 1', 'Band 2']
            # a = sns.pairplot(pd.DataFrame(MB_matrix,
            #                               columns=Bandnames),
            #                  diag_kind='kde', plot_kws={"s": 3})
            #
            # a.fig.suptitle("Pair plot of Band images")
            #
            # PCnames = ['PC 1', 'PC 2']
            # b = sns.pairplot(pd.DataFrame(PC,
            #                               columns=PCnames),
            #                  diag_kind='kde', plot_kws={"s": 3})
            #
            # b.fig.suptitle("Pair plot of PCs")
            #
            # plt.show()
            #

