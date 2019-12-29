import numpy as np 
import scipy.io as sio
from PIL import Image
import pdb
import math
# import tifffile as tiff
import matplotlib.pyplot as plt
from plot_code import geo_grid, geo_grid_2,plot_bar,geo_plot_point,plot_grid
from scipy import signal
from scipy import stats
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
# from ncread import time_filter
from scipy import ndimage
#from prettyplotlib import brewer2mpl
import pdb
import scipy.io as sio
from scipy.interpolate import griddata
import os
from scipy.ndimage.filters import maximum_filter1d
import pandas as pd
import glob
from skimage import measure
# from hw_blo import block_freq
from hw_blo_basic import geomap_china,block_freq
import time


def main_ta2hw():
	# calculate the occurrence of heatwave from temperature
	# for cpc tmp
	file_dir = '/home/user/Documents/research/project1'
	path = sorted(glob.glob(file_dir + '/cpc_tmp/ncdata/tmax*.nc'))
	print(path)
	ta_all = np.zeros([80,180,1],dtype = np.float32)
	for i in range(len(path)):
		nc = Dataset(path[i])
		ta = nc.variables['tmax'][:,:,:]
		lon = nc.variables['lon'][:]
		lat = nc.variables['lat'][:]
		print(ta.shape)
		ta = np.transpose(ta,[1,2,0]) # lat lon time 
		# ta = ta[30:110,500:680,:] # the europe
		ta = ta[30:110,140:320,:]
		ta_all = np.append(ta_all,ta,axis = 2)

	ta_all = ta_all[:,:,1:]
	ta_all[np.where(ta_all < -90)] = np.nan # 3 dimention

	ta_mean = np.nanmean(ta_all,axis = 2) # 
	clevs1 = np.arange(-5,20,5)
	title1 = 'ta_mean'
	outfig1_1 = file_dir + '/cpc_tmp/ta_mean_cpc_new.pdf'
	single_data = True


	tamax_dif_90,tamax_dif_90_de, tamax_25_summer,tamax_25_summer_de, tamax_75_summer,tamax_75_summer_de, tamax_ori_summer,tamax_detrend_summer= tmax_detrend(ta_all)

	hw_ori = hw_from_tamax(tamax_dif_90)
	hw_detrend = hw_from_tamax(tamax_dif_90_de)
	# np.savez(file_dir+ '/cpc_tmp/hw_cpc_3d.npy',hw_ori = hw_ori,hw_detrend = hw_detrend)

	hw_intens = hw_intensity(tamax_dif_90,tamax_25_summer,tamax_75_summer)
	hw_intens_detrend = hw_intensity(tamax_dif_90_de,tamax_25_summer_de,tamax_75_summer_de)


	np.savez(file_dir+ '/cpc_tmp/hw_intensity_cpc_3d_new.npy',hw_intens = hw_intens,hw_intens_detrend = hw_intens_detrend)





def tmax_detrend(tamax):
	mov_15 = True
	trend_remove_all = False
	times = pd.date_range('1979-01-01', '2017-12-31', name='time')
	year_num = times.year.max() - times.year.min()

	# remove the significant trend for each grid
	idx_summer, = np.where((times.month == 6) | (times.month == 7) | (times.month == 8))
	idx_summer_2d = np.reshape (idx_summer, [92,int(idx_summer.shape[0]/92)],order= 'F') # [92,39]
	idx_summer_lag7 = [range(idx_summer_2d[0,year]-7,idx_summer_2d[-1,year]+7) for year in range(39)] # each column is a year with 92 days

	tamax_summer_lag7 = tamax[:,:,idx_summer_lag7] # 80,180,39,105 # original nondetrended data
	tamax_summer_lag7 = np.transpose(tamax_summer_lag7,[0,1,3,2])
	tamax_summer_lag7 = np.reshape(tamax_summer_lag7,[80,180,39*105],order ='F')
	plt.plot(np.squeeze(tamax_summer_lag7[0,31,:]))
	# plt.show()
	if trend_remove_all:
		tamax_summer_lag7_detrend = signal.detrend(tamax_summer_lag7,axis = -1)
	else:
		tamax_summer_lag7_detrend = np.zeros([80,180,39*105],dtype= np.float32)
		tamax_summer_lag7_detrend[:] = np.nan
		for row in range(80):
			for col in range(180):
				year = np.array(range(39*105))
				sample_grid = np.squeeze(tamax_summer_lag7[row,col,:])
				mask, = np.where(~np.isnan(sample_grid))
				# pdb.set_trace()
				if mask.shape[0] > 20*105:
					print(mask.shape[0])
					slope, intercept, r_value, p_value, std_err = stats.linregress(year[mask],sample_grid[mask])
					if p_value < 0.05:
						sample_trend = year*slope
						# pdb.set_trace()
					else:
						sample_trend = 0
					tamax_summer_lag7_detrend[row,col,:] = sample_grid - sample_trend

	plt.plot(np.squeeze(tamax_summer_lag7_detrend[0,31,:]))
	# plt.show()
	# pdb.set_trace()
	tamax_summer_lag7_detrend_4d = np.reshape(tamax_summer_lag7_detrend,[80,180,105,39],order='F')
	tamax_summer_lag7_detrend_4d = np.transpose(tamax_summer_lag7_detrend_4d,[0,1,3,2])
	tamax_detrend = tamax.copy()
	tamax_detrend[:,:,idx_summer_lag7] = tamax_summer_lag7_detrend_4d
	# pdb.set_trace()

	tamax_win15d = np.zeros([80,180,92,39,15],dtype= np.float32)
	tamax_win15d_detrend = np.zeros([80,180,92,39,15],dtype= np.float32)
	idx_day = 0
	for mon in range(6,9): # pay attention please
		print(mon)
		for day in range(1,32):
			idx, = np.where((times.month == mon) & (times.day == day))
			if idx.shape[0]>0:
				idx_15 = [range(idx_day - 7, idx_day + 8) for idx_day in idx]
				tamax_win15d[:,:,idx_day,:,:] = tamax[:,:,idx_15]
				tamax_win15d_detrend[:,:,idx_day,:,:] = tamax_detrend[:,:,idx_15]
				idx_day = idx_day + 1
				print("idx_day",idx_day)

	tamax_win15d  = np.transpose(tamax_win15d, [4,3,0,1,2])
	tamax_win15d_detrend  = np.transpose(tamax_win15d_detrend, [4,3,0,1,2])
	
	tamax_win15d = np.reshape(tamax_win15d,[39*15,80,180,92],order = 'F')
	tamax_win15d_detrend = np.reshape(tamax_win15d_detrend,[39*15,80,180,92], order= 'F')

	tamax_90th_summer = np.nanpercentile(tamax_win15d,90,axis = 0)
	tamax_90th_summer_detrend = np.nanpercentile(tamax_win15d_detrend,90,axis = 0)

	tamax_25th_summer = np.nanpercentile(tamax_win15d,25,axis = 0)
	tamax_25th_summer_detrend = np.nanpercentile(tamax_win15d_detrend,25,axis = 0)

	tamax_75th_summer = np.nanpercentile(tamax_win15d,75,axis = 0)
	tamax_75th_summer_detrend = np.nanpercentile(tamax_win15d_detrend,75,axis = 0)


	# tamax_90th_summer_years =  np.repeat(tamax_90th_summer,39, axis = 2) # not repeat 
	# tamax_90th_summer_years_detrend =  np.repeat(tamax_90th_summer_detrend,39, axis = 2)

	tamax_90th_summer_years =  np.tile(tamax_90th_summer,(1,1,39)) # not repeat 
	tamax_90th_summer_years_detrend =  np.tile(tamax_90th_summer_detrend,(1,1,39)) 

	tamax_25th_summer_years =  np.tile(tamax_25th_summer,(1,1,39)) 
	tamax_25th_summer_years_detrend =  np.tile(tamax_25th_summer_detrend,(1,1,39)) 

	tamax_75th_summer_years = np.tile(tamax_75th_summer,(1,1,39)) 
	tamax_75th_summer_years_detrend =  np.tile(tamax_75th_summer_detrend,(1,1,39)) 

	tamax_ori_summer = tamax[:,:,idx_summer]
	tamax_detrend_summer = tamax_detrend[:,:,idx_summer]

	tamax_dif_90 = tamax_ori_summer - tamax_90th_summer_years
	tamax_dif_90_de = tamax_detrend_summer - tamax_90th_summer_years_detrend
	pdb.set_trace()

	return tamax_dif_90,tamax_dif_90_de, tamax_25th_summer_years,tamax_25th_summer_years_detrend, tamax_75th_summer_years,tamax_75th_summer_years_detrend,tamax_ori_summer, tamax_detrend_summer


def hw_from_tamax(ta):
	# heatwave from tamax
	# tamax dif from 90th
	ta_c = ta.copy()
	ta_c[np.where(ta_c > 0 )] = 1
	ta_c[np.where(ta_c <= 0 )] = 0
	hw = time_filter(ta_c,3)
	return hw

def hw_intensity(dif_90,ta_25th,ta_75th):
	# the difference between tmp and 90th
	dif_90[np.where(dif_90 < 0 )] = 0
	hw_intes = np.zeros_like(dif_90)
	hw_intes  = np.divide(dif_90 ,(ta_75th- ta_25th))

	return hw_intes


def time_filter(data,day_valid):
	# 3-dimention row col time
	data_new = np.zeros_like(data)
	for col in range(data.shape[1]):
		for row in range(data.shape[0]):
			sample = np.squeeze(data[row,col,:])
			mean_idx = np.convolve(sample, np.ones((day_valid,))/(day_valid), mode='valid') 	# longitude non-zero 15
			valid_idx, = np.where(mean_idx==1)
			if np.sum(valid_idx)>0:
				for idx in valid_idx:
					data_new[row,col,idx:idx+day_valid]=1
	return data_new

def time_filter_new(data,day_valid):
	# 4-dimention day year row col
	data_new = np.zeros_like(data)
	for col in range(data.shape[3]):
		for row in range(data.shape[2]):
			for year in range(data.shape[1]):
				sample = np.squeeze(data[:,year,row,col])
				mean_idx = np.convolve(sample, np.ones((day_valid,))/(day_valid), mode='valid') 	# longitude non-zero 15
				valid_idx, = np.where(mean_idx==1)
				if np.sum(valid_idx)>0:
					for idx in valid_idx:
						data_new[idx:idx+day_valid,year,row,col]=1
	return data_new


main_ta2hw()









