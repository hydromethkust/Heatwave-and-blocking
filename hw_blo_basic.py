import numpy as np 
import scipy.io as sio
from PIL import Image
import pdb
import math
import matplotlib.pyplot as plt

from plot_code import geo_grid, geo_grid_2,plot_bar,geo_plot_point,geo_grid_new2,geo_grid_sig,plot_code

from scipy import signal,stats,ndimage
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from skimage import measure
from prettyplotlib import brewer2mpl

from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
import statistics



def load_data():
	# pdb.set_trace()
	north_china = True
	file_dir = '/home/user/Documents/research/project1'
	# *************** china only: 0-55N 70-140E 110*140 ***************
	# hw_sum = sio.loadmat('ghwr_china_jja_day3_pdf.mat')['ghwr_china_jja']
	# hw_sum_3d = hw_sum.copy()
	# hw_sum = np.transpose(hw_sum,[2,0,1])
	# hw_sum = np.reshape(hw_sum,[92,39,110,140],order='F')
	# hw_sum[np.isnan(hw_sum)]= 0

	# hw_sum = sio.loadmat('./hw_data/hw_summer_1979_2017_20_90_70_160.mat')['hw'] # derived from the GHWR dataset
	hw_sum = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N.mat')['hw'] # derived from ERA-Interim 95th consecutive 3 days 

	# hw_sum = sio.loadmat('./hw_data/hw_event_area_thres_300_overlap_0.5_extent_50_duration_5_35_75N_3d.mat')['hw_evnt']
	# hw_sum = sio.loadmat('./hw_data/hw_event_area_thres_500_overlap_0.5_extent_50_duration_6_3d.mat')['hw_evnt']

	# hw_sum = hw_sum[30:110,:,:] # 35-75N # 70-160
	hw_sum_3d = hw_sum.copy()
	hw_sum = np.transpose(hw_sum,[2,0,1])
	hw_sum = np.reshape(hw_sum,[92,39,hw_sum_3d.shape[0],hw_sum_3d.shape[1]],order='F')
	# hw_sum[np.isnan(hw_sum)]= 0


 	# ******* small area ***************
	# data_sum = data_sum[:,:,30:50,:] # 30-40N
	# data_sum = data_sum[:,:,1:30,:]; # 40-55N


	# blo_sum = sio.loadmat('summer_blocking_spatial_filter_5day_1979_2017.mat')['blocking'] # TM 90 spatial-time filter
	# ************************************ in this study *********************************************************
	# blo_sum_tm90 = sio.loadmat(file_dir + '/blocking/summer_blocking_spatial_filter_5day_1979_2017_0_90N_valid.mat')['blocking'] # TM 90 0_90 N north 
	blo_sum_tm90 = sio.loadmat(file_dir + '/blocking/summer_blocking_nospatial_filter_5day_1979_2017_0_90N_valid.mat')['blocking']  # no spatial filter

	# blo_sum_weak = sio.loadmat('./era-pv/summer_china_daily_weak_blocking_0.7_5day_north_3di_new.mat')['blocking'] # pv weak <-0.7 5day
	# blo_sum_weak = sio.loadmat('./era-pv/summer_china_daily_weak_blocking_0.7_5day_20_70N_3di_weighted.mat')['blocking'] # 20-70 70-140
	# blo_sum_weak = sio.loadmat('./era-pv/weighted_10_levels/summer_china_daily_weak_blocking_0.7_5day_north_20_90_3di_weighted.mat')['blocking'] # pv_blocking 10 levels weighted
	
	# ****************** track algorithm ***************  
	# blo_sum_weak = sio.loadmat('./era-pv/data_source/_summer_north_asia_daily_blocking_pv1.2_5day_track_area_720_weighted_3d_all.mat')['blocking'] # pv_anom < -1.2 , 5day 720 grid track
	# blo_sum_weak = sio.loadmat('./era-pv/data_source/summer_north_asia_daily_blocking_pv1.2_5day_track_weighted_area_720_3d_all_new.mat')['blocking']
	# blo_sum_weak = sio.loadmat(file_dir + '/blocking/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_3d_all_35_75_valid_1_ratio_0.5.mat')['blocking'] ## check in 6h 
	# blo_sum_weak = sio.loadmat(file_dir + '/blocking_event/summer_north_asia_daily_blocking_pv1.0_5day_2dtrack_weighted_3d_all_35_75_valid_1_ratio_0.7_de3_0.5_6h_extent_100.mat')['blocking']
	# blo_sum_weak = sio.loadmat(file_dir + '/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_3d_all_35_75_valid_1_ratio_0.7_de3_0.5_6h_extent_100.mat')['blocking']
	# print(blo_sum_weak.shape)
	#*********************based on daily data*************************** 
	# blo_sum_weak = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.0_5day_2dtrack_weighted_3d_35_75N_all_ratio_0.35_daily_extent_100.mat')['blocking'] # check in daily
	blo_sum_weak = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_3d_35_75N_all_ratio_0.4_daily_extent_100.mat')['blocking']
	# blo_sum_weak = sio.loadmat('./era-pv/data_source/summer_north_asia_daily_blocking_pv0.7_5day_notrack_weighted_3d_all.mat')['blocking']
	
	# blo_sum =sio.loadmat('blocking_index_time_only_day3_summer.mat')['data_sum'] # TM90 index
	# blo_sum =sio.loadmat('./era-pv/summer_china_daily_intense_blocking_1.3_3day_3d.mat')['blocking'] # based on potential vorticity (intense)
	# blo_sum =sio.loadmat('./era-pv/summer_china_daily_weak_blocking_0.7_3day_3d.mat')['blocking'] # weak
	# blo_sum = sio.loadmat('./era-pv/summer_china_daily_abs_blocking_1_3day_3d.mat')['blocking'] # absolute 

	

	# blo_sum = sio.loadmat('./era-pv/summer_china_pv_1979_2017.mat')['vapv'] ### continious data 4 times a day
	# blo_sum = np.reshape(blo_sum, [4,92,39,blo_sum.shape[1],blo_sum.shape[2]],order = 'F')
	# blo_sum = np.squeeze(np.nanmean(blo_sum, axis =0))
	# blo_sum = np.transpose(blo_sum,[2,3,0,1]) # vapv only
	# blo_sum = np.reshape(blo_sum, [blo_sum.shape[0],blo_sum.shape[1],92*39],order = 'F') 

	# pdb.set_trace()
	# 0-90N 
	# blo_sum=blo_sum[30:-1,20:-40-1,:]  #nother 0.25 wester 0.25 top left point 
	# blo_sum = blo_sum[:-1,20:-40-1,:] # 0-90
	# pdb.set_trace()
	# blo_sum_tm90 = blo_sum_tm90[60:120,20:-40-1,:] # 30-60N, 60-160E
	# blo_sum_tm90 = blo_sum_tm90[30:140,20:-40-1,:] # 75-20N
	# blo_sum_tm90 = blo_sum_tm90[30:110,20:-40-1,:] # 75-20N
	# blo_sum_tm90 = blo_sum_tm90[30:110,20:-40-1,:] # 75-20N
	blo_sum_tm90 = blo_sum_tm90[30:110,20:-1,:] # 75-20N
	blo_sum_tm90_3d = blo_sum_tm90.copy()

	# blo_sum_weak = blo_sum_weak[60:120,20:-40-1,:] # 30-60N, 60-160E
	# blo_sum_weak = blo_sum_weak[20:80,:-1,:] # 30-60N, 60-160E
	# blo_sum_weak = blo_sum_weak[30:-1,:-1,:] # 70-20N
	# blo_sum_weak = blo_sum_weak[:-1,:-1,:]
	# blo_sum_weak = blo_sum_weak[30:-30-1,:-1,:]
	blo_sum_weak_3d = blo_sum_weak.copy() # no clip 20-90N

	# china_shape = tiff.imread('china_0.5_255.tif')
	# china_shape_copy = china_shape.copy()
	# china_shape = np.repeat(china_shape[:, :, np.newaxis],92*39,axis=2)

	 # only consider china
	# blo_sum[np.where(china_shape==255)] = 0 
	blo_sum_tm90 = np.transpose(blo_sum_tm90,[2,0,1])
	blo_sum_tm90 = np.reshape(blo_sum_tm90,[92,39,blo_sum_tm90.shape[1],blo_sum_tm90.shape[2]],order='F') # 4-dimention

	blo_sum_weak = np.transpose(blo_sum_weak,[2,0,1])
	blo_sum_weak = np.reshape(blo_sum_weak,[92,39,blo_sum_weak.shape[1],blo_sum_weak.shape[2]],order='F') # 4-dimention


	return  hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak

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

def lag_co_occur(hw,blo,lag0,lag1,exclusive_hw = None, exclusive_blo = None):
	# hw blo all 4-dimension
	# lag -- blocking happen before heatwave
	# exlusive_hw means one heatwave day can only be assigned once
	# exlusive_blo means one blockign day can only be assigned once
	if exclusive_hw == False and exclusive_blo == True:
		hw_and_blo_old = np.zeros_like(hw)
		for lag in range(lag0,lag1+1):
			if lag == 0:
				hw_new = hw.copy()
			else:
				# hw_lag = hw[:hw.shape[0]-lag,:,:,:] # wrong
				# hw_zero = np.zeros([lag,hw.shape[1], hw.shape[2],hw.shape[3]],dtype = np.float16)
				# hw_new   =  np.append(hw_zero, hw_lag,axis =0)
				hw_lag = hw[lag:,:,:,:]
				hw_zero = np.zeros([lag,hw.shape[1], hw.shape[2],hw.shape[3]],dtype = np.float16)
				hw_new   =  np.append(hw_lag,hw_zero,axis =0)
				
			hw_and_blo  =  co_occur(hw_new, blo,False,False)
			hw_and_blo_old[np.where(hw_and_blo==1)] = 1

		return hw_and_blo_old 
	
	elif exclusive_hw == True and exclusive_blo == True:
		hw_copy = hw.copy()
		hw_and_blo_old = np.zeros_like(hw)
		for lag in range(lag0,lag1+1):
			if lag == 0:
				hw_new = hw.copy()
				# hw_and_blo = co_occur(hw_copy,blo,False,False)
				# hw_copy[np.where(hw_and_blo==1)] = 0
				# hw_and_blo_old[np.where(hw_and_blo==1)]=1
			else:
				hw_lag = hw_copy[1:,:,:]
				hw_zero = np.zeros([lag,hw.shape[1], hw.shape[2],hw.shape[3]],dtype = np.float16)
				hw_new = np.append(hw_lag,hw_zero,axis =0)
				hw_copy = hw_new
				# hw_and_blo  =  co_occur(hw_new, blo,False,False)
				# hw_copy[np.where(hw_and_blo==1)] = 0
				# hw_and_blo_old[np.where(hw_and_blo==1)]=1
			hw_and_blo  =  co_occur(hw_new, blo,False,False)
			hw_copy[np.where(hw_and_blo==1)] = 0
			hw_and_blo_old[np.where(hw_and_blo==1)]=1

		return hw_and_blo_old

	elif exclusive_hw == True and exclusive_blo == False:
		hw_copy = hw.copy()
		hw_and_blo_old = np.zeros_like(hw)
		for lag in range(lag0,lag1+1):
			if lag == 0:
				blo_new = blo.copy()
			else:
				blo_lag = blo[:blo.shape[0]-lag,:,:,:]
				blo_zero = np.zeros([lag,blo.shape[1], blo.shape[2],blo.shape[3]],dtype = np.float16)
				blo_new   =  np.append(blo_zero,blo_lag,axis =0)

			hw_and_blo  =  co_occur(hw_copy, blo_new,False,False)
			hw_and_blo_old[np.where(hw_and_blo==1)] = 1

		return hw_and_blo_old 



def lon_mean_blocking(lon_inter,blocking_ratio,run_mean):
	# for a blocking 
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	if not run_mean:
		blo_weak_mean = np.zeros_like(blo_sum_weak)
		blo_tm90_mean = np.zeros_like(blo_sum_weak)

		blo_sum_weak_lon_sum = np.nansum(blo_sum_weak,axis = 2)  
		blo_sum_tm90_lon_sum = np.nansum(blo_sum_tm90,axis = 2)
		
		blo_sum_weak_lon_sum[np.where(blo_sum_weak_lon_sum < blocking_ratio)] = 0
		blo_sum_weak_lon_sum[np.where(blo_sum_weak_lon_sum > blocking_ratio)] = 1
		
		blo_sum_tm90_lon_sum[np.where(blo_sum_tm90_lon_sum < blocking_ratio)] = 0
		blo_sum_tm90_lon_sum[np.where(blo_sum_tm90_lon_sum > blocking_ratio)] = 1

		idx1_weak = np.where(blo_sum_weak_lon_sum==1)
		idx1_tm90 = np.where(blo_sum_tm90_lon_sum==1)
		blo_weak_mean[idx1_weak[0],idx1_weak[1],:,idx1_weak[2]] = 1
		blo_tm90_mean[idx1_tm90[0],idx1_tm90[1],:,idx1_tm90[2]] = 1
		# pdb.set_trace()
	return blo_weak_mean, blo_tm90_mean


def mon_blo_freq():
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	blo_sum  = blo_sum_tm90
	blo_mon_freq = np.zeros([3,39,blo_sum_weak.shape[2],blo_sum_weak.shape[3]],dtype = np.float16)
	for row in range(blo_sum_weak.shape[2]):
		print(row)
		for col in range(blo_sum_weak.shape[3]):
			for year in range(blo_sum_weak.shape[1]):
				blo_mon_freq[0,year,row,col] = np.nansum(np.squeeze(blo_sum[0:30,year,row,col]))/30
				blo_mon_freq[1,year,row,col] = np.nansum(np.squeeze(blo_sum[30:61,year,row,col]))/31
				blo_mon_freq[2,year,row,col] = np.nansum(np.squeeze(blo_sum[61:92,year,row,col]))/31
	
	blo_mon_freq  = np.reshape(blo_mon_freq,[39*3,80,180],order= 'F')

	sio.savemat('/home/user/Documents/research/project1/month_blocking_tm90_freq.mat',{'blo_mon':blo_mon_freq})


def make_ax(grid,data,outfig,outtitle):
	# lontitude; lat; time ; intensity
    fig = plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "serif"
    ax = fig.gca(projection='3d')
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
       
    ax.set_ylabel("Longitude")
    ax.set_xlabel("Latitude")
    ax.set_zlabel("Time")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.grid(grid)

    plt.title(outtitle)
    # ax.voxels(data, edgecolors='darkgray',facecolors = 'steelblue')
    
    # colors = np.chararray(data.shape,itemsize = 9)
    # colors[:] = 'steelblue'
    # colors[np.where(data>1)] = 'red'

    colors = np.zeros(data.shape,dtype = np.float16)
    colors  = colors.astype(str)
    colors[np.where(colors =='0.0')] = 'steelblue'
    colors[np.where(data>1)] = 'coral'
    # pdb.set_trace()
    ax.voxels(data, edgecolors='darkgray',facecolors = colors)

    ax.tick_params(labelsize = 12)
    plt.tight_layout()
    fig.savefig(outfig)
    # return ax

def daynum(data, grid_thres,interval, de_trend, thres,yearly = True):
	# data 4-dimension day year row col 
	# thres = True threshold 
	# else category 

	if yearly:
		sum_gph_daynum = np.zeros([data.shape[1]],dtype=np.float16)
		for year in range(data.shape[1]):
			sum_gph_daynum[year]=0
			for day in range(data.shape[0]):
				sum_gph_day = np.squeeze(data[day,year,:,:])
				if thres:
					if np.nansum(np.nansum(sum_gph_day)) > grid_thres:
						sum_gph_daynum[year] = sum_gph_daynum[year]+1
				else:
					if np.nansum(np.nansum(sum_gph_day)) > grid_thres and np.nansum(np.nansum(sum_gph_day)) <= (grid_thres+ interval):
						print(grid_thres,grid_thres+ interval)
						sum_gph_daynum[year] = sum_gph_daynum[year]+1
		if de_trend:
			sum_gph_daynum = signal.detrend(sum_gph_daynum)
	else:
		sum_gph_daynum = np.zeros([data.shape[1],3],dtype=np.float16)
		for year in range(data.shape[1]):
			sum_gph_daynum[year,:]=0
			for day in range(0,30):
				sum_gph_day = np.squeeze(data[day,year,:,:])
				if thres:
					if np.nansum(np.nansum(sum_gph_day)) > grid_thres:
						sum_gph_daynum[year,0] = sum_gph_daynum[year,0]+1
				else:
					if np.nansum(np.nansum(sum_gph_day)) > grid_thres and np.nansum(np.nansum(sum_gph_day)) <= (grid_thres+ interval):
						print(grid_thres,grid_thres+ interval)
						sum_gph_daynum[year,0] = sum_gph_daynum[year,0]+1
			for day in range(30,61):
				sum_gph_day = np.squeeze(data[day,year,:,:])
				if thres:
					if np.nansum(np.nansum(sum_gph_day)) > grid_thres:
						sum_gph_daynum[year,1] = sum_gph_daynum[year,1]+1
				else:
					if np.nansum(np.nansum(sum_gph_day)) > grid_thres and np.nansum(np.nansum(sum_gph_day)) <= (grid_thres+ interval):
						print(grid_thres,grid_thres+ interval)
						sum_gph_daynum[year,1] = sum_gph_daynum[year,1]+1
			for day in range(61,92):
				sum_gph_day = np.squeeze(data[day,year,:,:])
				if thres:
					if np.nansum(np.nansum(sum_gph_day)) > grid_thres:
						sum_gph_daynum[year,2] = sum_gph_daynum[year,2]+1
				else:
					if np.nansum(np.nansum(sum_gph_day)) > grid_thres and np.nansum(np.nansum(sum_gph_day)) <= (grid_thres+ interval):
						print(grid_thres,grid_thres+ interval)
						sum_gph_daynum[year,2] = sum_gph_daynum[year,2]+1

		sum_gph_daynum = np.reshape(sum_gph_daynum,[data.shape[1]*3],order = 'F')

	return	sum_gph_daynum

def gridnum(data,day_thres,interval,de_trend,thres,yearly = True):
	# data 4-dimension day year row col 
	if yearly:
		sum_gph_gridnum = np.zeros(data.shape[1],dtype=np.float16)
		for year in range(data.shape[1]):
			data_sum_year = np.squeeze(data[:,year,:,:])
			sum_gph_grid = np.nansum(data_sum_year,axis=0) ## sum??
			# pdb.set_trace()
			if thres:
				sum_gph_grid[np.where(sum_gph_grid <= day_thres)] = 0
				sum_gph_grid[np.where(sum_gph_grid > day_thres)] = 1

			else:
				idx = np.where((sum_gph_grid > day_thres) & (sum_gph_grid <= day_thres+ interval)) # np.where not and but &
				non_idx =  np.where((sum_gph_grid <= day_thres) | (sum_gph_grid > day_thres+ interval))
				sum_gph_grid[non_idx] = 0
				sum_gph_grid[idx] = 1

			sum_gph_gridnum[year] = np.sum(np.sum(sum_gph_grid))

		if de_trend:
			sum_gph_gridnum = signal.detrend(sum_gph_gridnum)

	else:
		sum_gph_gridnum = np.zeros([data.shape[1],3],dtype=np.float16)
		for year in range(data.shape[1]):
			data_sum_year6 = np.squeeze(data[0:30,year,:,:])
			data_sum_year7 = np.squeeze(data[30:61,year,:,:])
			data_sum_year8 = np.squeeze(data[61:92,year,:,:])

			sum_gph_grid6 = np.nansum(data_sum_year6,axis=0) ## sum??
			sum_gph_grid7 = np.nansum(data_sum_year7,axis=0) ## sum??
			sum_gph_grid8 = np.nansum(data_sum_year8,axis=0) ## sum??
			if thres:
				sum_gph_grid6[np.where(sum_gph_grid6 <= day_thres)] = 0
				sum_gph_grid6[np.where(sum_gph_grid6 > day_thres)] = 1
				sum_gph_grid7[np.where(sum_gph_grid7 <= day_thres)] = 0
				sum_gph_grid7[np.where(sum_gph_grid7 > day_thres)] = 1
				sum_gph_grid8[np.where(sum_gph_grid8 <= day_thres)] = 0
				sum_gph_grid8[np.where(sum_gph_grid8 > day_thres)] = 1

			else:
				idx = np.where((sum_gph_grid6 > day_thres) & (sum_gph_grid6 <= day_thres+ interval)) # np.where not and but &
				non_idx =  np.where((sum_gph_grid6 <= day_thres) | (sum_gph_grid6 > day_thres+ interval))
				sum_gph_grid6[non_idx] = 0
				sum_gph_grid6[idx] = 1

			sum_gph_gridnum[year,0] = np.sum(np.sum(sum_gph_grid6))
			sum_gph_gridnum[year,1] = np.sum(np.sum(sum_gph_grid7))
			sum_gph_gridnum[year,2] = np.sum(np.sum(sum_gph_grid8))

		sum_gph_gridnum = np.reshape(sum_gph_gridnum,[data.shape[1]*3],order='F')

	return sum_gph_gridnum

def daily_gridnum(data):
	# data 4-dimension day year row col 
	sum_gph_daily_gird = np.zeros([data.shape[0],data.shape[1]],dtype=np.int32)
	for year in range(data.shape[1]):
		for day in range(data.shape[0]):
			sum_gph_day = np.squeeze(data[day,year,:,:])
			sum_gph_daily_gird[day,year] = np.nansum(np.nansum(sum_gph_day))
	return sum_gph_daily_gird

def persistence(data):
	# data 4-dimension day year row col remain check
	sum_gph_perst_all = np.zeros([13,data.shape[1]],dtype=np.float16)
	sum_gph_perst  = np.zeros([data.shape[0]+1,data.shape[1],data.shape[2],data.shape[3]],dtype=np.float16)

	for year in range(data.shape[1]):
		for row in range(data.shape[2]):
			for col in range(data.shape[3]):
				sum_ghp_year_grid = np.squeeze(data[:,year,row,col])
				idx0, =np.where(sum_ghp_year_grid==0)
				# pdb.set_trace()
				if idx0[-1] < 91:
					idx0 = np.append(idx0,92)
				idx1= np.append(idx0,0)
				idx0= np.insert(idx0,1,0) # wrong should be idx0 =  np.insert(idx0,0,0)
				perst = idx1-idx0-1
				perst[np.where(perst<3)]=0

				# if long_perst:
				# 	lon_perst, = np.where(perst>=5)
				# 	long_start = idx0[lon_perst]
				# 	long_end = idx1[lon_perst]
				# 	grid_lonperest = np.zeros([data.shape[0]], dtype=np.float16)  
				# 	for eve in range(lon_perst.shape[0]):
				# 		grid_lonperest[long_start[eve+1]:long_end[eve-1]]=1
				# 	sum_long_perst[:,year,row,col] = grid_lonperest
				# 	pdb.set_trace()

				sum_gph_perst[:idx0.shape[0],year,row,col] = perst
		perst_year = np.squeeze(sum_gph_perst[:,year,:,:])
		for daynum in range(3,15):
			idx = perst_year[np.where(perst_year==daynum)] # multi-dimention array return row,col...
			sum_gph_perst_all[daynum-3,year]=idx.shape[0]
	return sum_gph_perst_all, sum_gph_perst


# def grid_event(data):
# 	for year in range(data.shape[1]):
# 		for row in range(data.shape[2]):
# 			for col in range(data.shape[3]):
# 				sample = np.squeeze(data[:,year,row,col])
# 				idx0, = np.where(sample == 0)
# 				persis_sample = [idx0[i]-idx0[i-1] for i in range(1,idx0.shape[0]+1)]

def event_number(data, hw = None,tm = None, pv = None):
	# data 4dimenition data
	# consider the median length of event
	ev_num_grid = np.zeros([data.shape[2],data.shape[3]],dtype = np.float16)
	ev_duration_grid = np.zeros([data.shape[2],data.shape[3]],dtype = np.float16)
	ev_num_grid_year  = np.zeros([data.shape[2],data.shape[3],data.shape[1]],dtype = np.float16)
	if hw == True:
		d_thres = 2
	if tm == True:
		d_thres = 4
	if pv == False:
		for row in range(data.shape[2]):
			print(row)
			for col in range(data.shape[3]):
				ev_number  = 0
				ev_len = np.zeros([1],dtype = np.float16)
				for year in range(data.shape[1]):
					idx0, = np.where( np.squeeze(data[:,year,row,col]) == 0)
					event_length_ori = [idx0[i]-idx0[i-1]-1 for i in range(1,idx0.shape[0])]
					event_length_ori = [e for e in event_length_ori if e > d_thres]
					event_number_year = len(event_length_ori)
					# ev_len.append(event_length_ori)
					ev_len = np.append(ev_len,np.asarray(event_length_ori),axis = 0)
					ev_num_grid_year[row,col,year] = event_number_year
					ev_number = ev_number + event_number_year
				ev_len = ev_len[1:]
				ev_num_grid [row,col] = ev_number
				# pdb.set_trace()
				ev_duration_grid [row,col] = np.median(ev_len)
	if pv == True:
		print('pv')
		for row in range(data.shape[2]):
			for col in range(data.shape[3]):
				ev_number  = 0
				ev_len = np.zeros([1],dtype = np.float16)
				for year in range(data.shape[1]):
					sample = np.squeeze(data[:,year,row,col])
					unique_sample = np.unique(sample)
					unique_sample = np.delete(unique_sample, np.where(unique_sample==0)[0])
					event_number_year = unique_sample.shape[0]
					event_length_ori = [np.where(sample == idx)[0].shape[0] for idx in unique_sample]
					# pdb.set_trace()
					ev_len = np.append(ev_len,np.asarray(event_length_ori),axis = 0)
					ev_num_grid_year[row,col,year] = event_number_year
					ev_number = ev_number + event_number_year
				# pdb.set_trace()
				ev_len = ev_len[1:]
				ev_num_grid [row,col] = ev_number
				ev_duration_grid [row,col] = np.median(ev_len)

	return ev_num_grid,ev_duration_grid,ev_num_grid_year


def block_freq(data,dimen):
	# 2-dimention each grid
	# data 4-dimension day year row col 
	if dimen == 'grid':
		block_freq_= np.zeros([data.shape[2],data.shape[3]],dtype=np.float16)
		for row in range(data.shape[2]):
			for col in range(data.shape[3]):
				block_freq_[row,col] = np.nansum(np.squeeze(data[:,:,row,col])) 
	elif dimen == 'time':
		block_freq_= np.zeros([data.shape[0],data.shape[1]],dtype=np.float16)
		for day in range(data.shape[0]):
			for year in range(data.shape[1]):
				block_freq_[day,year] = np.nansum(np.squeeze(data[day,year,:,:]))
	return  block_freq_

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def pv_overlap_check(data,over_ratio,time_interval,extent):
	# ******** data 3-dimention day row col, already labeled*******
	len_thres = 5 
	if time_interval == '6h':
		end_time = 91*4+3
	else:
		end_time = 91
	hw_sample = data[:,:,:].copy()
	idx_length = list()
	for label in np.unique(hw_sample):#range(1,int(hw_sample.max()+1)):
		idx_label = np.where(hw_sample ==label)
		hw_evnt_label = hw_sample.copy()
		hw_evnt_label[np.where(hw_evnt_label!=label)] = 0
		hw_evnt_label[np.where(hw_evnt_label==label)] = 1

		hw_evnt_sum = np.nansum(hw_evnt_label,axis = 0) 
		hw_evnt_daily_sum = np.nansum(np.squeeze(np.nansum(hw_evnt_label,axis = 1)),axis = 1)
		idx, = np.where(hw_evnt_daily_sum > 0)
		# print("idx_length is, idx:", idx.shape[0],idx)
		escape = False
		if idx.shape[0]>1: # only consider the label with overlap main happen
			# print("label is, label_length is ", label, np.unique(idx_label[0]).shape[0])
			idx_length.append(np.unique(idx_label[0]).shape[0])
			# hw_evnt_plot = np.transpose(hw_evnt_label,(1,2,0))
			# ax = make_ax(True)
			# ax.voxels(hw_evnt_plot, edgecolors='gray')
			# plt.show()
			# pdb.set_trace()
			for day in range(idx.min(),idx.max()): 
				# print("current day",day)
				if escape == True:
					escape = False
					continue
				# print("escape = ",escape)
				if day < 91*4+3:
					idx_day_label = np.where(np.squeeze(hw_evnt_label[day+1,:,:]) > 0)
				else:
					idx_day_label = []
				hw_evnt_daily_0 = np.squeeze(hw_evnt_label[day,:,:]).copy()
				# pdb.set_trace()
				if day < 91*4+3:
					hw_evnt_daily_1 = np.squeeze(hw_evnt_label[day+1,:,:]).copy()
				else:
					hw_evnt_daily_1 = np.squeeze(hw_evnt_label[day,:,:]).copy()
				hw_day1_area_ori = np.sum(hw_evnt_daily_1[np.where(hw_evnt_daily_1>0)])
				hw_evnt_daily_1[np.where(hw_evnt_daily_0==0)] = 0 # co-located area
				
				hw_day0_area = np.sum(hw_evnt_daily_0[np.where(hw_evnt_daily_0>0)])
				hw_day1_area = np.sum(hw_evnt_daily_1[np.where(hw_evnt_daily_1>0)])
				# print("area_day0,area_day1,area_co:",hw_day0_area,hw_day1_area_ori,hw_day1_area)
				if day < 91*4+3:
					if hw_day0_area > 0 and (hw_day1_area < over_ratio * hw_day0_area):
						# print("******area_day0,area_day1,rario******",hw_day0_area,hw_day1_area,hw_day1_area/hw_day0_area)
						# print("*****idx_day_label*****",idx_day_label)
						hw_evnt_label[day+1,idx_day_label[0],idx_day_label[1]] = 0
						data[day+1,idx_day_label[0],idx_day_label[1]] = 0
						escape = True # jump to 2 days later
					# if hw_day0_area ==0:
					# 	hw_evnt_label[day+1,idx_day_label[0],idx_day_label[1]] = 0
					# 	data[day+1,idx_day_label[0],idx_day_label[1]] = 0

					if hw_day0_area < extent:
						idx_day_label = np.where(np.squeeze(hw_evnt_label[day,:,:]) > 0)
						hw_evnt_label[day,idx_day_label[0],idx_day_label[1]] = 0
						data[day,idx_day_label[0],idx_day_label[1]] = 0
	
	# print("idx_length is shown as follow",idx_length)

	return data

def pv_overlap_check_2d(data,over_ratio,over_ratio_1_0,time_interval,extent,neigh):
	# data 4 dimension day year row col
	# time_interval 6hr or daily
	# neigh = 4 or 6
	data_feature = np.zeros([92*39*4*10,10],dtype = np.int32)
	coords_list = list()
	event_num = 0   
	labeled_hw = np.zeros(data.shape)
	for year in range(data.shape[1]):
		for day in range(data.shape[0]):
			sample = np.squeeze(data[day,year,:,:]).copy()
			labeled_sample = measure.label(sample,neighbors = neigh)
			for region in measure.regionprops(labeled_sample):
    			#take regions with large enough areas
				if region.area < extent:
					labeled_sample[region.coords[:,0],region.coords[:,1]] = 0
			labeled_sample[np.where(labeled_sample >0)] = 1
			labeled_sample_new = measure.label(labeled_sample,neighbors = neigh)
			loca_0 = np.where(labeled_sample_new==0)
			# labeled_hw[day,year,:,:] = labeled_sample_new + day*10
			labeled_hw[day,year,:,:] = labeled_sample_new + day*1000
			labeled_hw[day,year,loca_0[0],loca_0[1]]= 0
			for region in measure.regionprops(labeled_sample_new):
				data_feature[event_num,0]= year
				data_feature[event_num,1]= day
				# data_feature[event_num,2]= region.label + 10*day
				data_feature[event_num,2]= region.label + 1000*day
				data_feature[event_num,3]= region.area
				data_feature[event_num,4]= region.bbox[0]
				data_feature[event_num,5]= region.bbox[1]
				data_feature[event_num,6]= region.bbox[2]
				data_feature[event_num,7]= region.bbox[3]
				data_feature[event_num,8]= region.centroid[0]
				data_feature[event_num,9]= region.centroid[1]
				coords_list.append(region.coords)
				event_num = event_num + 1
	data_feature  = np.delete(data_feature,range(len(coords_list),data_feature.shape[0]),axis= 0)
	event_df =  pd.DataFrame(data_feature, columns = ['year','day','label','area','min_row','min_col','max_row','max_col','cen_row','cen_col'])
	print("length of coords", len(coords_list))
	# pdb.set_trace()
	# label the event satisfy the overlap check with same label not only 0 and 1
	# label_new = 10
	for event in range(len(coords_list)):
		# print(event)
		day_all = event_df.day
		year_all = event_df.year

		day_0 = event_df.day[event]
		year_0 = event_df.year[event]
		
		day_0_label = event_df.label[event]
		# print("day_0_label",day_0_label) 
		day_0_coords = coords_list[event]
		day_0_coords_copy = list((day_0_coords[i][0],day_0_coords[i][1]) for i in range(day_0_coords.shape[0]))

		day_1_idx = np.where((day_all == day_0+1) & (year_all == year_0))

		if len(day_1_idx[0])>0:
			for i in day_1_idx[0]:
				day_1_coords = coords_list[i]
				day_1_coords_copy = list((day_1_coords[i][0],day_1_coords[i][1]) for i in range(day_1_coords.shape[0]))
				day_0_1_intersection = list(set(day_0_coords_copy) & set(day_1_coords_copy))
				overlap_ratio = len(day_0_1_intersection) / len(day_0_coords_copy)
				overlap_ratio_1_0 = len(day_1_coords_copy) / len(day_0_coords_copy)
				# print(overlap_ratio)
				part_labeled_hw = np.squeeze(labeled_hw[:day_0+1,year_0,:,:])
				idx_0 = np.where(part_labeled_hw==day_0_label) # the length of the label_0
				# pdb.set_trace()

				### ****************** if only based on the timestep****************** 
				# if np.unique(idx_0[0]).shape[0]< 3:
				# 	if overlap_ratio > over_ratio:
				# 		# print("event num", "overlasp_ratio",event, overlap_ratio)
				# 		labeled_hw[int(day_0+1),int(year_0),day_1_coords[:,0],day_1_coords[:,1]] = day_0_label
				# 		event_df.label[i] =  int(day_0_label)
				# 		if overlap_ratio_1_0 >  over_ratio_1_0:
				# 			labeled_hw[int(day_0),int(year_0),day_0_coords[:,0],day_0_coords[:,1]] = 0 # change the lower level label into 0 

				# elif np.unique(idx_0[0]).shape[0]> 2:
				# 	# print(day_0,day_0_label,np.unique(idx_0[0]).shape[0])
				# 	# *********** constant extent *************** 
				# 	# if len(day_0_1_intersection) > extent:
				# 	# 	labeled_hw[int(day_0+1),int(year_0),day_1_coords[:,0],day_1_coords[:,1]] = day_0_label
				# 	# 	event_df.label[i] =  int(day_0_label)
				# 	# smaller ratio
				# 	if overlap_ratio > 0.7:
				# 		labeled_hw[int(day_0+1),int(year_0),day_1_coords[:,0],day_1_coords[:,1]] = day_0_label
				# 		event_df.label[i] =  int(day_0_label)
				# 		if overlap_ratio_1_0 >  over_ratio_1_0:
				# 			labeled_hw[int(day_0),int(year_0),day_0_coords[:,0],day_0_coords[:,1]] = 0 # change the lower level label into 0 

				##### ****************** if based on the slope ************************* 
				if overlap_ratio > over_ratio:
					# print("event num", "overlasp_ratio",event, overlap_ratio)
					labeled_hw[int(day_0+1),int(year_0),day_1_coords[:,0],day_1_coords[:,1]] = day_0_label
					event_df.label[i] =  int(day_0_label)
					if overlap_ratio_1_0 >  over_ratio_1_0:
						labeled_hw[int(day_0),int(year_0),day_0_coords[:,0],day_0_coords[:,1]] = 0 # change the lower level label into 0 
				# elif idx_0[0].shape[0]>2 and (np.where(idx_0[0]== idx_0[0][-1])[0].shape[0]) < (np.where(idx_0[0]== idx_0[0][-1]-1)[0].shape[0]) and (np.where(idx_0[0]== idx_0[0][-1]-1)[0].shape[0]) < (np.where(idx_0[0]== idx_0[0][-1]-2)[0].shape[0]):
				# 	if overlap_ratio > 0.5:
				# 		# pdb.set_trace()
				# 		labeled_hw[int(day_0+1),int(year_0),day_1_coords[:,0],day_1_coords[:,1]] = day_0_label
				# 		event_df.label[i] =  int(day_0_label)
				# 		if overlap_ratio_1_0 >  over_ratio_1_0:
				# 			labeled_hw[int(day_0),int(year_0),day_0_coords[:,0],day_0_coords[:,1]] = 0 # change the lower level label into 0 

	return labeled_hw




def pv_thres(data,area_thres, len_thres):
	file_dir = '/home/user/Documents/research/project1'
	'''***** data 3-dimention for a specific single year: day row col
	to check if the blocking event achieve the spatil-temporal threshold'''

	for region in measure.regionprops(data):
		labeled_hw_sample_copy = data.copy()
		labeled_hw_sample_copy[np.where(labeled_hw_sample_copy!=region.label)]=0
		label_idx = np.where(data == region.label) 
		hw_sum_2d = np.sum(labeled_hw_sample_copy,axis=0)
		extent = np.where(hw_sum_2d>0)[0].shape[0]
		# length = region.bbox[3]- region.bbox[0] # length
		length = np.unique(label_idx[0]).shape[0]
		# print(length,region.area,region.label)
		if extent < area_thres or length < len_thres:
			# print("removed label,extent, length", region.label,extent,length)
			data[np.where(data==region.label)] = 0
		# print("area of label 48", np.where(data==48)[0].shape[0])
	print("data_element",np.unique(data))

	return data

def co_occur(data_hw,data_blo,neigh_time,neigh_grid):
	# 4- dimeantion
	data_blo_copy = data_blo.copy()
	# pdb.set_trace()
	if neigh_time == True:
		data_blo_copy1 = data_blo.copy()
		# have a lag for +-1 days
		for year in range(data_blo.shape[1]):
			for row in range(data_blo.shape[2]):
				for col in range(data_blo.shape[3]):
					sample =  np.squeeze(data_blo_copy1[:,year,row,col])
					# pdb.set_trace()
					idx_1, = np.where(sample ==1)
					idx_be = idx_1 -1
					idx_af = idx_1 +1
					idx_all = np.concatenate((idx_1,idx_be,idx_af),axis=0)
					# pdb.set_trace()
					idx_all= np.unique(idx_all)
					idx_nan = np.where(np.logical_or(idx_all<0,idx_all>sample.shape[0]-1))
					idx_all = np.delete(idx_all,idx_nan)
					print(idx_all)
					# pdb.set_trace()
					data_blo_copy1[idx_all,year,row,col] =1 

		data_blo_copy[np.where(data_blo_copy1==1)] = 1


	if neigh_grid == True:
		data_blo_copy2 = data_blo.copy()
		for day in range(data_blo.shape[0]):
			for year in range(data_blo.shape[1]):
				blo_map = data_blo_copy2[day,year,:,:]
				for row in range(blo_map.shape[0]):
					print("row=",row)
					blo_map_copy = blo_map.copy()
					# pdb.set_trace()
					sample = np.squeeze(blo_map_copy[row,:])
					idx_1, = np.where(sample ==1)
					idx_be = idx_1 -1
					idx_af = idx_1 +1
					idx_all = np.concatenate((idx_1,idx_be,idx_af),axis=0)
					idx_all= np.unique(idx_all)
					# pdb.set_trace()
					idx_nan = np.where(np.logical_or(idx_all < 0, idx_all > sample.shape[0]-1))
					idx_all = np.delete(idx_all,idx_nan)
					print(idx_all)
					blo_map_copy[row,idx_all] =1
				for col in range(blo_map.shape[1]):
					sample = np.squeeze(blo_map[:,col])
					idx_1, = np.where(sample ==1)
					idx_be = idx_1 -1
					idx_af = idx_1 +1
					idx_all = np.concatenate((idx_1,idx_be,idx_af),axis=0)
					idx_all= np.unique(idx_all)
					idx_nan = np.where(np.logical_or(idx_all < 0, idx_all > sample.shape[0]-1))
					idx_all = np.delete(idx_all,idx_nan)
					print(idx_all)
					blo_map[idx_all,col] =1
					blo_map[np.where(blo_map_copy==1)]=1
			data_blo_copy2[day,year,:,:] = blo_map
		data_blo_copy[np.where(data_blo_copy2==1)] = 1

		# any dimention
	data_hw_copy =  data_hw.copy()
	data_hw_copy[np.where(data_blo_copy ==0)]=0

	return data_hw_copy

def condi_prob(co_occur_data, data, dimen):
	# data 4-dimension day year row col 
	if dimen == 'grid' :
		con_prob = np.zeros([data.shape[2],data.shape[3]],dtype = np.float16)
		for row in range(data.shape[2]):
			for col in range(data.shape[3]):
				con_prob[row,col] = np.nansum(np.squeeze(co_occur_data[:,:,row,col]))/np.nansum(np.squeeze(data[:,:,row,col]))
				# condi_prob_[row,col] = np.divide(np.nansum(np.squeeze(co_occur_data[:,:,row,col])),np.nansum(np.squeeze(data[:,:,row,col])))
	elif dimen == 'time':
		con_prob = np.zeros([data.shape[0],data.shape[1]],dtype = np.float16)
		for day in range(data.shape[0]):
			for year in range(data.shape[1]):
				# condi_prob_[row,col] = np.nansum(np.squeeze(co_occur_data[day,year,:,:]))/np.nansum(np.squeeze(data[day,year,:,:]))
				con_prob[row,col] = np.divide(np.nansum(np.squeeze(co_occur_data[day,year,:,:])),np.nansum(np.squeeze(data[day,year,:,:])))

	return con_prob


def cor_relation_2d(data_hw,data_blo,order):
	coe_rp = np.zeros([data_hw.shape[1],data_blo.shape[1],2],dtype=np.float16)
	for col1 in range(data_hw.shape[1]):
		for col2 in range(data_blo.shape[1]):
			if order:
				coe_rp[col1,col2,0], coe_rp[col1,col2,1] = sp.stats.kendalltau(data_hw[:,col1],data_blo[:,col2])
				# pdb.set_trace()
			else:
			 slope, intercept, coe_rp[col1,col2,0], coe_rp[col1,col2,1], std_err = linregress(data_hw[:,col1],data_blo[:,col2]) 
	return coe_rp

def cor_relation_1d(data):
	cor_kp = np.zeros([data.shape[1],2],dtype = np.float16)
	# x_data = range(1979,2018)
	x_data = range(data.shape[0])
	for col in range(data.shape[1]):
		cor_kp[col,0], intercept, r, cor_kp[col,1], std_err = linregress(x_data,data[:,col])
	return cor_kp

# def geomap_china(data1, data2, data3,clevs1,clevs2,clevs3,out_title,outfig, single_data):
# def geomap_china(data1, data2, data3,data4,clevs1,clevs2,clevs3,clevs4,out_title,outfig, single_data):
def geomap_china(data1, data2,clevs1,clevs2,title1,title2,outfig,single_data,sig = None,sig_level = 0.05): #two figure share one colorbar
# def geomap_china(data1, data2,clevs1,clevs2,out_title,outfig, single_data): #single data
	# china shape 
	# china_shape = tiff.imread('china_0.5_255.tif')
	# china_shape_copy = china_shape.copy()
	# china_shape_copy = china_shape_copy[:-20,:]
	# china_shape_copy = np.flipud(china_shape_copy)

	print("row= ", data1.shape[0])
	print("col=", data1.shape[1])

	lllon = 70 + 0.25 
	lllat = 35 + 0.25
	
	urlon = 160 - 0.25
	urlat = 75 - 0.25

	print("lat row= ", (urlat-lllat+0.5)*2)
	print("lon col=", (urlon-lllon+0.5)*2)
	
	lon_0 = (lllon+urlon)/2
	lat_0 = (lllat+urlat)/2


	# since the code in the plot_code did upside-down the latitude
	data1 = np.flipud(data1)
	data2 = np.flipud (data2)

	# data1[np.where(china_shape_copy==255)]=np.nan
	data1[np.where(data1 >= clevs1.max())] = clevs1.max()
	data1[np.where(data1 <= clevs1.min())] = clevs1.min()


	# data2[np.where(china_shape_copy==255)]=np.nan
	data2[np.where(data2 >= clevs2.max())] = clevs2.max()
	data2[np.where(data2 <= clevs2.min())] = clevs2.min()

	if single_data:
		# pdb.set_trace()
		geo_grid(data1,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs1,title1,outfig)
	if not single_data:
		if not sig:
		# geo_grid_2(data1,data2,data3,data4,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs1,clevs2,clevs3,clevs4,out_title,outfig) # data1-contour line; data_2:filled contour
		# geo_grid_2(data1,data2,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs1,clevs2,out_title,outfig)
			geo_grid_new2(data1,data2,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs1,title1,title2,outfig)
		if sig:
			geo_grid_sig(data1,data2,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs1,title1,title2,outfig,sig_level) # data1:grid data; data2:point data


# lon_mean_blocking(0,20,False)

# mon_blo_freq()