import numpy as np 
import scipy.io as sio
from PIL import Image
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import tifffile as tiff
from plot_code import geo_grid, geo_grid_2,plot_bar,geo_plot_point,plot_cdf,tsplot
from plot_code import plot_grid # plot_grid(data,outfig, vmax_v,vmin_v):
from hw_blo_basic import time_filter, time_filter_new
from scipy import signal,stats,ndimage
from scipy.stats import linregress, entropy,gaussian_kde,zscore
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from skimage import measure
from prettyplotlib import brewer2mpl
import glob
# import pyKstest as kstest2
from mlxtend.plotting import ecdf
from  pyKstest import kstest2 
from operator import itemgetter
import itertools
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from hw_blo_basic import pv_overlap_check,pv_overlap_check_2d,pv_thres  # for the big event detection
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from hw_blo_basic import load_data,mon_blo_freq, make_ax, daynum, gridnum, daily_gridnum, persistence, block_freq, co_occur, condi_prob, cor_relation_2d, cor_relation_1d, geomap_china,lon_mean_blocking,lag_co_occur,event_number


''' To identify the heatwave event; seperate BRH and BURH; quantify their features'''

def main_hw_event():
	track_2d = False
	file_dir = '/home/user/Documents/research/project1'


	# ta_90th = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N_detrend_105_39_sig_only.mat')['hw']
	ta_90th = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_detrend']
	ta_90th = np.transpose(ta_90th,[2,0,1])
	ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')
	
	hw_grid = ta_90th

	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
	for day in range(hw_grid.shape[0]):
		for year in range(hw_grid.shape[1]):
			hw_grid[day,year,:,:][np.where(land_mask == 0 )] = 0

	hw_track = np.zeros(hw_grid.shape,dtype = np.float32)
	
	if track_2d:
		# hw_grid_track = pv_overlap_check_2d(hw_grid,over_ratio = 0.4,over_ratio_1_0 = 5, time_interval='1d',extent = 0, neigh = 4)
		hw_grid_track = pv_overlap_check_2d(hw_grid,over_ratio = 0.4,over_ratio_1_0 = 100, time_interval='1d',extent = 0, neigh = 4)
		pdb.set_trace()

		# sio.savemat(file_dir + '/heatwave_event/hw_event_cpc_detrend_sig_track_daily_or_0.4_100.mat',{'hw_event':hw_grid_track})
		np.save(file_dir + '/heatwave_event/hw_event_cpc_detrend_sig_track_daily_or_0.4_100.npy',hw_grid_track)

	# hw_grid_track = sio.loadmat(file_dir + '/heatwave_event/hw_event_cpc_detrend_sig_track_daily_or_0.4_100.mat')['hw_event'] # land_only
	hw_grid_track = np.load(file_dir + '/heatwave_event/hw_event_cpc_detrend_sig_track_daily_or_0.4_100.npy') # land_only

	year_num = 39
	for year in range(year_num):
		print("year = ", year)
		if track_2d:
			hw_sample = np.squeeze(hw_grid_track[:,year,:,:]).copy()
		else:
			hw_sample = np.squeeze(hw_grid_track[:,year,:,:]).copy()  # ????

		labeled_hw_sample = measure.label(hw_sample, neighbors = 4)

		for region in measure.regionprops(labeled_hw_sample):
    		#take regions with large enough areas
			if region.area < 3:
				labeled_hw_sample[region.coords.T[0],region.coords.T[1],region.coords.T[2]] = 0

		
		hw_thres_sample = pv_thres(labeled_hw_sample,area_thres = 0,len_thres = 3)  # area_thres 1.8*10^6 km2 720 400 1.2 track extent: for each time step 

		###### **************************** plot event ************************ 
		# # hw_out = measure.label(hw_thres_sample, neighbors = 4) # ***
		# for region in measure.regionprops(hw_thres_sample):
		# 	if region.area > 2:
		# 		# ***************** no temporal/spatial projection **********************
		# 		hw_evnt_plot = hw_thres_sample[region.bbox[0]:region.bbox[3],region.bbox[1]:region.bbox[4],region.bbox[2]:region.bbox[5]].copy()  # hw_thres_sample??
		# 		hw_evnt_plot[np.where(hw_evnt_plot!= region.label)] = 0
		# 		hw_evnt_plot_ori = np.transpose(hw_evnt_plot,[1,2,0]) # row col time
		# 		hw_evnt_plot_new = hw_evnt_plot_ori.copy()
		# 		# hw_evnt_plot_new[10:19,0:6,:] = 2
		# 		# hw_evnt_plot_new[13:20,15:20,1:] = 3
		# 		# hw_evnt_plot_new[np.where(hw_evnt_plot_ori==0)] = 0
		# 		# hw_evnt_plot_new[np.where(hw_evnt_plot_new==1)] = 0
		# 		# hw_evnt_plot_new[1:5,3:10,:] = 2
		# 		# pdb.set_trace()
				
		# 		# # ***********with projection******************
		# 		# hw_evnt_plot = hw_thres_sample[region.bbox[0]:region.bbox[3]+5,region.bbox[1]-3:region.bbox[4],region.bbox[2]-3:region.bbox[5]+3].copy()  # hw_thres_sample??
		# 		# hw_evnt_plot[np.where(hw_evnt_plot!= region.label)] = 0
		# 		# hw_evnt_plot_ori = np.transpose(hw_evnt_plot,[1,2,0]) # row col time
		# 		# hw_evnt_plot_new1 = np.zeros_like(hw_evnt_plot_ori)
		# 		# for time in range(hw_evnt_plot_ori.shape[2]-5):
		# 		# 	hw_evnt_plot_new1[:,:,time+5] = hw_evnt_plot_ori[:,:,time]
		# 		# hw_evnt_plot_new1[:,:,:3] = 0
		# 		# hw_evnt_plot_new = hw_evnt_plot_new1.copy()
		# 		# hw_evnt_plot_new[10:19,0:6,6:] = 2
		# 		# hw_evnt_plot_new[13:20,15:20,6:] = 3
		# 		# hw_evnt_plot_new[np.where(hw_evnt_plot_new1==0)] = 0
		# 		# # hw_evnt_plot_new[np.where(hw_evnt_plot_ori==0)] = 0
		# 		# # hw_evnt_plot_new[np.where(hw_evnt_plot_new==1)] = 0

		# 		outfig = file_dir + '/heatwave_event/figure/area_50/ratio_0.4/cpc_ad_new_detrend_heatwave_output_event_year_' + str(year+1979) + '_area_'+ str(region.area) + '_label_' + str(region.label) + '.png'
		# 		# outtitle = 'heatwave_output_event_year_' + str(year+1979) + '_area_'+ str(region.area) + '_label_' + str(region.label) 
		# 		outtitle = 'Heatwave_event'
		# 		# pdb.set_trace()
		# 		make_ax(True,hw_evnt_plot_new,outfig,outtitle)
		# 		plt.show()
				# pdb.set_trace()
		hw_track[:,year,:,:] = hw_thres_sample

	# sio.savemat(file_dir + '/heatwave_event/hw_event_area_thres_3_overlap_0.35_extent_0_duration_3_35_75N_4d_land_only_labeled_detrended_sig_only.mat',{'hw_evnt':hw_track})
	# sio.savemat(file_dir + '/heatwave_event/hw_event_area_thres_3_overlap_0.4_extent_0_duration_3_35_75N_4d_land_only_labeled_nondetrended.mat',{'hw_evnt':hw_track})
	np.save(file_dir + '/cpc_tmp/hw_event_area_thres_3_overlap_0.4_extent_0_duration_3_35_75N_4d_land_only_labeled_detrended.npy',hw_track)
	pdb.set_trace()

	# hw_track_3d = np.reshape(hw_track,[92*39,hw_track.shape[2],hw_track.shape[3]],order = 'F')
	# hw_track_3d = np.transpose(hw_track_3d,[1,2,0])
	# # pdb.set_trace()
		
	# # pdb.set_trace()
	# # sio.savemat('./hw_blo_ta/hw_event_area_thres_300_overlap_0.5_extent_50_duration_5_35_75N_land_only.mat',{'hw_evnt':hw_track})
	# sio.savemat('./hw_blo_ta/hw_event_area_thres_0_overlap_0.5_extent_0_duration_3_35_75N_3d_land_only_valid.mat',{'hw_evnt':hw_track_3d})
	# sio.savemat('./hw_blo_ta/hw_event_area_thres_0_overlap_0.5_extent_0_duration_3_35_75N_4d_land_only_valid.mat',{'hw_evnt':hw_track})


def main_hw_blo_evnt():
	file_dir = '/home/user/Documents/research/project1'
	# blo_evnt = sio.loadmat('./era-pv/data_source/summer_north_asia_daily_blocking_pv1.2_5day_track_weighted_area_720_4d_all_new.mat')['blocking']
	# blo_evnt = sio.loadmat('./era-pv/data_source/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_4d_all_35_75_valid_1_ratio_0.5.mat')['blocking']
	# blo_evnt = sio.loadmat('./era-pv/data_source/summer_north_asia_daily_blocking_pv1.0_5day_2dtrack_weighted_4d_all_35_75_valid_1_ratio_0.5_6h.mat')['blocking']
	
	# blo_evnt = sio.loadmat(file_dir+'/blocking_event/summer_north_asia_daily_blocking_pv1.0_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.35_daily_extent_100.mat')['blocking']
	
	# blo_evnt = blo_evnt[:,:,30:110,:-1]
	# hw_evnt = sio.loadmat('./hw_data/hw_event_area_thres_300_overlap_0.5_extent_50_duration_5_35_75N.mat')['hw_evnt']

	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	blo_evnt = blo_sum_tm90
	# blo_evnt = blo_sum_weak

	# hw_evnt = sio.loadmat('./hw_blo_ta/hw_event_area_thres_300_overlap_0.5_extent_50_duration_5_35_75N_land_only.mat')['hw_evnt']
	# hw_evnt = sio.loadmat('./hw_blo_ta/hw_event_area_thres_0_overlap_0.5_extent_0_duration_3_35_75N_3d_land_only_valid.mat')['hw_evnt']
	# hw_evnt = np.transpose (hw_evnt,[2,0,1])
	# hw_evnt = np.reshape(hw_evnt,[92,39,80,180],order = 'F')

	## labeled by check_2d method 
	# hw_evnt = sio.loadmat('./hw_blo_ta/hw_event_area_thres_0_overlap_0.5_extent_0_duration_3_35_75N_4d_land_only_labeled_detrended_mean.mat')['hw_evnt']
	# hw_evnt = sio.loadmat(file_dir + '/heatwave_event/hw_event_area_thres_3_overlap_0.45_extent_0_duration_3_35_75N_4d_land_only_labeled_detrended_sig_only.mat')['hw_evnt']
	
	''' initial submission'''
	# hw_evnt = sio.loadmat(file_dir + '/heatwave_event/hw_event_area_thres_3_overlap_0.4_extent_0_duration_3_35_75N_4d_land_only_labeled_nondetrended.mat')['hw_evnt']
	
	''' first revision '''
	hw_evnt = np.load(file_dir + '/cpc_tmp/hw_event_area_thres_3_overlap_0.4_extent_0_duration_3_35_75N_4d_land_only_labeled_nondetrended.npy')
	hw_evnt = hw_evnt.astype(np.int16)


	'''intial submisson '''
	# hw_intens = sio.loadmat(file_dir + '/temperature/hw_tamax_90_intensity_3d_25_75th.mat')['hw_intes']
	# hw_intens = sio.loadmat(file_dir + '/temperature/hw_tamax_90_intensity_3d_75_25th_detrend_sig_only.mat')['hw_intes']

	''' first revision '''
	hw_intens = np.load(file_dir + '/cpc_tmp/hw_intensity_cpc_3d_new.npz')['hw_intens']
	hw_intens = np.transpose(hw_intens,[2,0,1])
	hw_intens = np.reshape(hw_intens,[92,39,80,180],order ='F')

	# hw_evnt = hw_evnt[:,:,30:110,:]
	neigh_time = False
	neigh_grid = False
	# pdb.set_trace()
	# hw_and_blo = co_occur(hw_evnt,blo_evnt,neigh_time,neigh_grid)
	hw_feature = list()
	hw_blo_feature = list()
	# hw_labeled = np.zeros(hw_evnt.shape, dtype = np.int16)


	hw_feature_array = np.zeros([10000,17]) #  year hw: area length extent  hw and blo: area length extent and their coresponding ratio centrid label intensity of heatwave
	event_num = 0
	hw_tamax = list()

	# add the tamax time series for each event.... 

	for year in range(39):
		print("year = ", year)
		year_real = year +1979
		# ****************** relabel the 2d_check dataset with 3d method*************** 

		# # hw_blo_sample = np.squeeze(hw_and_blo[:,year,:,:]).copy()
		# hw_sample = np.squeeze(hw_evnt[:,year,:,:]).copy()
		# labeled_hw_sample = measure.label(hw_sample, neighbors = 4)
		# # hw_labeled[:,year,:,:] = labeled_hw_sample
		# print(labeled_hw_sample.max())

		# ************* use the 2d labeled dataset *****************
		labeled_hw_sample =  np.squeeze(hw_evnt[:,year,:,:])

		hw_intens_sample = np.squeeze(hw_intens[:,year,:,:]).copy()
		blo_sample = np.squeeze(blo_evnt[:,year,:,:]).copy()



		# hw_label_list = list()
		# hw_area_list = list()
		# hw_length_lisio.savemat(file_dir + '/heatwave_event/all_hw_event_feature_ratio_0.3_area_3_duration_3_extend_0_35_75N_with_label_pv1.0_land_only_labeled_2d_detrended_sig_only.mat',{'hw_blo_feature':hw_feature_array})st = list()
		# hw_extent_list = list()

		labeled_hw_sample_copy = labeled_hw_sample.copy()

		labeled_hw_sample_copy2 = labeled_hw_sample.copy()
		labeled_hw_sample_copy2[np.where(blo_sample == 0)] = 0


		for region in measure.regionprops(labeled_hw_sample):
			print(region.label)
			labeled_hw_sample_copy = labeled_hw_sample.copy()
			hw_intens_sample_copy = hw_intens_sample.copy()
			# # *********** plot ************ 
			# hw_evnt_plot = labeled_hw_sample[region.bbox[0]:region.bbox[3],region.bbox[1]:region.bbox[4],region.bbox[2]:region.bbox[5]].copy()
			# hw_evnt_plot = np.transpose(hw_evnt_plot,[1,2,0]) # row col time
			# outfig = 'hw_1995_check_input_event_area_' + str(region.area) + '_label_' + str(region.label) + '.png' 
			# outtitle = 'hw_1995_check_input_event_area_' + str(region.area) + '_label_' + str(region.label) 
			# make_ax(True, hw_evnt_plot,outfig,outtitle)
			# plt.show()

			hw_feature_array[event_num,0] = year_real
			hw_feature_array[event_num,1] = event_num + 1
			hw_feature_array[event_num,2] = region.area
			hw_feature_array[event_num,3] = region.bbox[3]- region.bbox[0] # length

			labeled_hw_sample_copy[np.where(labeled_hw_sample_copy!=region.label)]=0
			hw_intens_sample_copy[np.where(labeled_hw_sample_copy!=region.label)] = 0

			hw_sum_2d = np.sum(labeled_hw_sample_copy,axis=0)/region.label
			hw_sum_2d_0 = hw_sum_2d[np.where(hw_sum_2d >0)]
			hw_length_75th = np.percentile(hw_sum_2d_0,75)
			
			hw_length_25th = np.percentile(hw_sum_2d_0,25)
			hw_extent_75th = np.where(hw_sum_2d >= hw_length_25th)[0].shape[0]

			hw_feature_array[event_num,4] = np.where(hw_sum_2d>0)[0].shape[0] # extent



			labeled_hw_sample_copy3 = labeled_hw_sample_copy2.copy()
			label_idx = np.where(labeled_hw_sample_copy3 == region.label) 
			labeled_hw_sample_copy3[np.where(labeled_hw_sample_copy3!=region.label)] = 0
			hw_feature_array[event_num,5] = label_idx[0].shape[0] # area
			labeled_sum = np.sum(labeled_hw_sample_copy3,axis =0)
			hw_feature_array[event_num,6] = np.unique(label_idx[0]).shape[0] # lengthfa
			hw_feature_array[event_num,7] = np.array(np.where(labeled_sum>0))[0].shape[0] # extent

			hw_feature_array[event_num,8]= np.divide(hw_feature_array[event_num,5],hw_feature_array[event_num,2]) # ratio of area 
			hw_feature_array[event_num,9]= np.divide(hw_feature_array[event_num,6],hw_feature_array[event_num,3])
			hw_feature_array[event_num,10]= np.divide(hw_feature_array[event_num,7],hw_feature_array[event_num,4])

			row_bbox = region.bbox[1]
			col_bbox = region.bbox[2]

			hw_feature_array[event_num,11] = region.centroid[1]
			hw_feature_array[event_num,12] = region.centroid[2]
			hw_feature_array[event_num,13] = region.label
			hw_feature_array[event_num,14] = np.sum(hw_intens_sample_copy)
			hw_feature_array[event_num,15] = hw_length_25th
			hw_feature_array[event_num,16] = hw_extent_75th

			# pdb.set_trace()
			event_num = event_num + 1

		# pdb.set_trace()


	
	# sio.savemat('./era-pv/data_source/hw_event_feature_area_300_duration_5_extend10_35_75N_with_label.mat',{'hw_feature':hw_feature_array})
	# sio.savemat(file_dir + '/heatwave_event/all_hw_event_feature_ratio_0.35_area_3_duration_3_extend_0_35_75N_with_label_tm90_nospatial_filter_daily_land_only_labeled_2d_detrended_sig_only.mat',{'hw_blo_feature':hw_feature_array})
	'''initial submssion '''
	# sio.savemat(file_dir + '/heatwave_event/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_0.4_daily_land_only_labeled_2d_nondetrended_sig_only.mat',{'hw_blo_feature':hw_feature_array})
	
	''' first revision '''
	# np.save(file_dir + '/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_0.4_daily_land_only_labeled_2d_nondetrended.npy',hw_feature_array)
	np.save(file_dir + '/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_tm90_nospatial_filter_daily_land_only_labeled_2d_nondetrended_new.npy',hw_feature_array)
	pdb.set_trace()


# def main_blo_related_hw_ev(): orginal source function
def main_hw_blo_event_lag():

	''' the influence of blocking on the characteristics of heatwave '''
	file_dir = '/home/user/Documents/research/project1'
	lag0 = -7
	lat1 = 7
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	blo_evnt = blo_sum_tm90
	# pv_label = sio.loadmat(file_dir +'/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.4_daily_extent_100_with_label.mat')['blocking']
	# blo_evnt = pv_label

	''' inital  submission'''
	# hw_evnt = sio.loadmat(file_dir + '/heatwave_event/hw_event_area_thres_3_overlap_0.4_extent_0_duration_3_35_75N_4d_land_only_labeled_detrended_sig_only.mat')['hw_evnt']

	'''first revision'''
	hw_evnt = np.load(file_dir + '/cpc_tmp/hw_event_area_thres_3_overlap_0.4_extent_0_duration_3_35_75N_4d_land_only_labeled_detrended.npy')
	hw_evnt = hw_evnt.astype(np.int16)

	# hw_intens = sio.loadmat(file_dir + '/temperature/hw_tamax_90_intensity_3d_25_75th.mat')['hw_intes']
	''' initial submission'''
	# hw_intens = sio.loadmat(file_dir + '/temperature/hw_tamax_90_intensity_3d_75_25th_detrend_sig_only.mat')['hw_intes']
	''' first revision'''
	hw_intens = np.load(file_dir + '/cpc_tmp/hw_intensity_cpc_3d_new.npz')['hw_intens_detrend']

	hw_intens = np.transpose(hw_intens,[2,0,1])
	hw_intens = np.reshape(hw_intens,[92,39,80,180],order ='F')
	hw_feature_array = np.zeros([10000,17])
	
	# if only sart-end -7~7 based on heatwave event
	event_num = 0
	for year in range(39):
		print("year = ", year)
		year_real = year +1979
		
		# ************* use the 2d labeled dataset *****************
		labeled_hw_sample =  np.squeeze(hw_evnt[:,year,:,:])
		hw_intens_sample = np.squeeze(hw_intens[:,year,:,:]).copy()
		blo_year = np.squeeze(blo_evnt[:,year,:,:]).copy()

		labeled_hw_sample_copy = labeled_hw_sample.copy()
		# labeled_hw_sample_copy2 = labeled_hw_sample.copy()
		# labeled_hw_sample_copy2[np.where(blo_sample == 0)] = 0

		for region in measure.regionprops(labeled_hw_sample):
			print(region.label)
			coords_2d = np.unique(region.coords[:,1:],axis =0)
			if region.bbox[0]>6:
				event_start =  region.bbox[0] - 7
			else:
				event_start = 0
			if region.bbox[3] > 84:
				event_end = 91
			else:
				event_end = region.bbox[3] + 7
			blo_sample  = blo_year[event_start:event_end+1,coords_2d[:,0],coords_2d[:,1]]
			# pdb.set_trace()
			labeled_hw_sample_copy = labeled_hw_sample.copy()
			hw_intens_sample_copy = hw_intens_sample.copy()
			# # *********** plot ************ 
			# hw_evnt_plot = labeled_hw_sample[region.bbox[0]:region.bbox[3],region.bbox[1]:region.bbox[4],region.bbox[2]:region.bbox[5]].copy()
			# hw_evnt_plot = np.transpose(hw_evnt_plot,[1,2,0]) # row col time
			# outfig = 'hw_1995_check_input_event_area_' + str(region.area) + '_label_' + str(region.label) + '.png' 
			# outtitle = 'hw_1995_check_input_event_area_' + str(region.area) + '_label_' + str(region.label) 
			# make_ax(True, hw_evnt_plot,outfig,outtitle)
			# plt.show()
			hw_feature_array[event_num,0] = year_real
			hw_feature_array[event_num,1] = event_num + 1
			hw_feature_array[event_num,2] = region.area
			hw_feature_array[event_num,3] = region.bbox[3]- region.bbox[0] # length

			labeled_hw_sample_copy[np.where(labeled_hw_sample_copy!=region.label)]=0
			hw_intens_sample_copy[np.where(labeled_hw_sample_copy!=region.label)] = 0

			hw_sum_2d = np.sum(labeled_hw_sample_copy,axis=0)/region.label
			# hw_sum_2d_0 = hw_sum_2d[np.where(hw_sum_2d >0)]
			# hw_length_75th = np.percentile(hw_sum_2d_0,75)
			# hw_length_25th = np.percentile(hw_sum_2d_0,25)
			# hw_extent_75th = np.where(hw_sum_2d >= hw_length_25th)[0].shape[0]
			hw_feature_array[event_num,4] = np.where(hw_sum_2d>0)[0].shape[0] # extent
			
			blo_extent_idx, = np.where(np.nansum(blo_sample,axis = 0)>0)
			blo_length_idx, = np.where(np.nansum(blo_sample,axis = 1)>0)
			# pdb.set_trace()
			blo_vol_idx = np.where(blo_sample>0)
			hw_feature_array[event_num,5] = blo_vol_idx[0].shape[0] # area
			print(np.nansum(np.where(blo_sample>0)))
			hw_feature_array[event_num,6] = blo_length_idx.shape[0] # length
			hw_feature_array[event_num,7] = blo_extent_idx.shape[0]  # extent
			
			hw_feature_array[event_num,8]= np.divide(hw_feature_array[event_num,5],blo_sample.shape[0]*blo_sample.shape[1]) # ratio of area 
			hw_feature_array[event_num,9]= np.divide(hw_feature_array[event_num,6],blo_sample.shape[0])
			hw_feature_array[event_num,10]= np.divide(hw_feature_array[event_num,7],blo_sample.shape[1])

			row_bbox = region.bbox[1]
			col_bbox = region.bbox[2]

			hw_feature_array[event_num,11] = region.centroid[1]
			hw_feature_array[event_num,12] = region.centroid[2]
			hw_feature_array[event_num,13] = region.label
			hw_feature_array[event_num,14] = np.sum(hw_intens_sample_copy)
			# hw_feature_array[event_num,15] = hw_length_25th
			# hw_feature_array[event_num,16] = hw_extent_75th
			event_num = event_num + 1
			# pdb.set_trace()
	
	''' initial submission'''
	# sio.savemat(file_dir + '/heatwave_event/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_tm90_daily_land_only_labeled_2d_detrended_sig_only_blo_lag7.mat',{'hw_blo_feature':hw_feature_array})
	''' first revision '''
	np.save(file_dir + '/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_tm90_daily_land_only_labeled_2d_detrended_sig_only_blo_lag7.npy',hw_feature_array)
	# np.save(file_dir + '/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_with_label_daily_land_only_labeled_2d_detrended_sig_only_blo_lag7.npy',hw_feature_array)



def main_blo_related_hw():
	duration_cate = False
	sub_region = True
	
	plotecdf = True
	ksresult = False

	DEOR = False
	volume = False
	deor_mean   = True

	blo_type = 'tm90_nospatial_filter'
	# blo_type = 'pv1.2_0.4_daily'
	hw_type = '0.4'
	area_thres = '0'
	detrend_type = 'detrend_sig_only'
	file_dir = '/home/user/Documents/research/project1'
	# hw_blo_fea = sio.loadmat(file_dir +'/heatwave_event/feature/all_hw_event_feature_ratio_0.5_area_3_duration_3_extend_0_35_75N_with_label_pv1.0_land_only_labeled_2d_detrended_sig_only.mat')['hw_blo_feature']
	# hw_blo_fea = sio.loadmat(file_dir +'/heatwave_event/feature/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_0.4_daily_land_only_labeled_2d_detrended_sig_only.mat')['hw_blo_feature']
	# hw_blo_fea = sio.loadmat(file_dir +'/heatwave_event/feature/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_tm90_nospatial_filter_daily_land_only_labeled_2d_detrended_sig_only.mat')['hw_blo_feature']
	# hw_blo_fea = sio.loadmat(file_dir +'/heatwave_event/feature/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_tm90_nospatial_filter_daily_land_only_labeled_2d_detrended_sig_only.mat')['hw_blo_feature']
	
	''' initial submission'''
	# hw_blo_fea = sio.loadmat(file_dir +'/heatwave_event/feature/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_0.4_daily_land_only_labeled_2d_detrended_sig_only_blo_lag7.mat')['hw_blo_feature']
	# hw_blo_fea = sio.loadmat(file_dir +'/heatwave_event/feature/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_tm90_daily_land_only_labeled_2d_detrended_sig_only_blo_lag7.mat')['hw_blo_feature']
	
	''' first revision '''
	hw_blo_fea = np.load(file_dir +'/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_tm90_daily_land_only_labeled_2d_detrended_sig_only_blo_lag7.npy')
	# hw_blo_fea = np.load(file_dir +'/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_with_label_daily_land_only_labeled_2d_detrended_sig_only_blo_lag7.npy')


	idx_0= np.where(hw_blo_fea[:,0]==0)
	hw_blo_fea = np.delete(hw_blo_fea,idx_0,axis=0)
	# pdb.set_trace()
	hw_blo_df =  pd.DataFrame(hw_blo_fea, columns = ['year','event_num','area','length','extent','co_area','co_length','co_extent','co_area_ratio','co_length_ratio','co_extent_ratio','center_row','center_col','hw_label','hw_intensity','hw_length_75th','hw_extent_75th'])
	
	# hw_area = hw_blo_df.area
	# hw_length = hw_blo_df.length
	# # pdb.set_trace()

	# if duration_cate:
	# 	hw_length [np.where(hw_length.values < 5)[0]] = 1
	# 	hw_length [np.where(hw_length.values > 4)[0]] = 2

	# hw_extent = hw_blo_df.extent
	# hw_intensity = hw_blo_df.hw_intensity
	# hw_length_75th = hw_blo_df.hw_length_75th
	# hw_extent_75th = hw_blo_df.hw_extent_75th


	# hw_blo_area = hw_blo_df.co_area
	# hw_blo_length = hw_blo_df.co_length
	# hw_blo_extent = hw_blo_df.co_extent

	# hw_blo_area_ratio = hw_blo_df.co_area_ratio
	# hw_blo_length_ratio = hw_blo_df.co_length_ratio
	# hw_blo_extent_ratio = hw_blo_df.co_extent_ratio
	# # pdb.set_trace()

	# ******** based on area ******** 
	# thres = 0.5
	# idx_blo, = np.where(hw_blo_area_ratio > thres) 
	# idx_num = idx_blo.shape[0]
	# idx_non_blo, = np.where(hw_blo_area_ratio <= thres)
	# idx_non_num = idx_non_blo.shape[0]

	# *********** ratio or constant ******************  
	# thres = [0.0,0.2,0.4,0.6,0.8]
	
	if plotecdf:
		thres = [0.2]
		region_row = [(0,40),(40,60),(60,80)] # divide the whole region into 3 sub-regions 75-55 55-45 45-35
		area_ks = np.zeros([5,3,4],dtype = np.float32)
		length_ks = np.zeros([5,3,4],dtype = np.float32)
		extent_ks = np.zeros([5,3,4],dtype = np.float32)
		intensity_ks = np.zeros([5,3,4],dtype = np.float32)
	
	if ksresult:
		thres = [0.0,0.2,0.4,0.6,0.8]
		# thres= [0,0.1,0.2,0.3,0.4,0.5]
		region_row = [(0,40),(40,60),(60,80)] # divide the whole region into 3 sub-regions 75-55 55-45 45-35
		area_ks = np.zeros([len(thres),3,4],dtype = np.float32)
		length_ks = np.zeros([len(thres),3,4],dtype = np.float32)
		extent_ks = np.zeros([len(thres),3,4],dtype = np.float32)
		intensity_ks = np.zeros([len(thres),3,4],dtype = np.float32)

	for i in range(len(thres)):
		len_thres = thres[i]
		extent_thres = thres[i]
		hw_area = hw_blo_df.area
		hw_length = hw_blo_df.length

		if duration_cate:
			hw_length [np.where(hw_length.values < 5)[0]] = 1
			hw_length [np.where(hw_length.values > 4)[0]] = 2

		hw_extent = hw_blo_df.extent
		hw_intensity = hw_blo_df.hw_intensity
		hw_length_75th = hw_blo_df.hw_length_75th
		hw_extent_75th = hw_blo_df.hw_extent_75th

		hw_center = hw_blo_df.center_row


		hw_blo_area = hw_blo_df.co_area
		hw_blo_length = hw_blo_df.co_length
		hw_blo_extent = hw_blo_df.co_extent

		hw_blo_area_ratio = hw_blo_df.co_area_ratio
		hw_blo_length_ratio = hw_blo_df.co_length_ratio
		hw_blo_extent_ratio = hw_blo_df.co_extent_ratio

		# ********** if delete the small event ************ 
		idx_nan, = np.where(hw_area < 0)
		hw_area = np.delete(hw_area.values, idx_nan, axis = 0)
		hw_length = np.delete(hw_length.values, idx_nan)
		hw_extent = np.delete(hw_extent.values, idx_nan)
		hw_intensity = np.delete(hw_intensity.values, idx_nan)
		hw_length_75th = np.delete(hw_length_75th.values, idx_nan)
		hw_extent_75th = np.delete(hw_extent_75th.values, idx_nan)
		hw_center = np.delete(hw_center.values, idx_nan)

		hw_blo_length_ratio = np.delete(hw_blo_length_ratio.values, idx_nan)
		hw_blo_extent_ratio = np.delete(hw_blo_extent_ratio.values, idx_nan)
		# pdb.set_trace()

		if plotecdf:
			if sub_region:
				# region = ['high','mid','low']
				region = ['high']
				for reg in range(len(region)):
					deor_mean = True
					if deor_mean:
						idx_blo, = np.where(((hw_blo_length_ratio+hw_blo_extent_ratio)/2 > thres[i]) & (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))
						idx_non_blo, = np.where(((hw_blo_length_ratio+hw_blo_extent_ratio)/2 <= thres[i]) & (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))
					else:
						idx_blo, = np.where((hw_blo_length_ratio > len_thres) & (hw_blo_extent_ratio > extent_thres) & (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))
						idx_non_blo, = np.where(((hw_blo_length_ratio <= len_thres) | (hw_blo_extent_ratio < extent_thres))& (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))
					idx_num = idx_blo.shape[0]
					idx_non_num = idx_non_blo.shape[0]
					# pdb.set_trace()

					hw_area_blo_related = hw_area[idx_blo]
					hw_length_blo_related = hw_length[idx_blo]
					hw_extent_blo_related = hw_extent[idx_blo]

					hw_area_nonblo_related = hw_area[idx_non_blo]
					hw_length_nonblo_related = hw_length[idx_non_blo]
					hw_extent_nonblo_related = hw_extent[idx_non_blo]

					hw_intensity_blo_related = hw_intensity[idx_blo]
					hw_intensity_nonblo_related = hw_intensity[idx_non_blo]

					hw_length_75th_blo_related = hw_length_75th[idx_blo]
					hw_length_75th_nonblo_related = hw_length_75th[idx_non_blo]
					hw_extent_75th_blo_related = hw_extent_75th[idx_blo]
					hw_extent_75th_nonblo_related = hw_extent_75th[idx_non_blo]
					# pdb.set_trace()
					# area_ks[i,reg,:2] = stats.ks_2samp(hw_area_blo_related,hw_area_nonblo_related) # 2-sides
					# length_ks[i,reg,:2] = stats.ks_2samp(hw_length_blo_related,hw_length_nonblo_related)
					# extent_ks[i,reg,:2] = stats.ks_2samp(hw_extent_blo_related,hw_extent_nonblo_related)
					# intensity_ks[i,reg,:2] = stats.ks_2samp(hw_intensity_blo_related,hw_intensity_nonblo_related)

					# ********** use kstest****************
					# area_ks[i,reg,:2] = kstest2(hw_area_blo_related,hw_area_nonblo_related)[1:] # one side # wrong
					# length_ks[i,reg,:2] = kstest2(hw_length_blo_related,hw_length_nonblo_related)[1:]
					# extent_ks[i,reg,:2] = kstest2(hw_extent_blo_related,hw_extent_nonblo_related)[1:]
					# intensity_ks[i,reg,:2] = kstest2(hw_intensity_blo_related,hw_intensity_nonblo_related)[1:]
					# area_ks[i,reg,2:] = idx_num,idx_non_num
					# length_ks[i,reg,2:] = idx_num,idx_non_num
					# extent_ks[i,reg,2:] = idx_num,idx_non_num
					# intensity_ks[i,reg,2:] = idx_num,idx_non_num


					# nbins = 100
					fig = plt.figure(figsize =(6,5))
					plt.rcParams["font.family"] = "serif"
					ax, _, _ = ecdf(hw_area_blo_related, x_label=None)
					ax, _, _ = ecdf(hw_area_nonblo_related, ax=ax,y_label=None)
					ax.tick_params(labelsize = 13)
					# plt.xlabel('HWV',fontsize = 12)
					plt.title('(e) HWV [TM index]',fontsize = 13)

					plt.ylabel('ECDF',fontsize = 13)
					plt.legend(['BRH', 'BURH'],prop={'size': 13})
					plt.show()
					fig.savefig(file_dir+ '/paper_figure/cpc_a_HWV_0.2_pv_north_hw_0.4_lag7_tm.pdf')

					fig = plt.figure(figsize =(6,5))
					plt.rcParams["font.family"] = "serif"
					ax, _, _ = ecdf(hw_extent_blo_related, x_label=None)
					ax, _, _ = ecdf(hw_extent_nonblo_related, ax=ax,y_label=None)
					ax.tick_params(labelsize = 13)
					plt.title('(c) HWE [TM index]',fontsize = 13)
					plt.ylabel('ECDF',fontsize = 13)
					plt.legend(['BRH', 'BURH'],prop={'size': 13})
					plt.show()
					fig.savefig(file_dir+ '/paper_figure/cpc_a_HWE_0.2_pv_north_hw_0.4_lag7_tm.pdf')


					fig = plt.figure(figsize =(6,5))
					plt.rcParams["font.family"] = "serif"
					ax, _, _ = ecdf(hw_length_blo_related, x_label=None)
					ax, _, _ = ecdf(hw_length_nonblo_related, ax=ax,y_label=None)
					ax.tick_params(labelsize = 13)
					plt.title('(a) HWD [TM index]',fontsize = 13)
					plt.ylabel('ECDF',fontsize = 13)
					plt.legend(['BRH', 'BURH'],prop={'size': 13})
					plt.show()
					fig.savefig(file_dir+ '/paper_figure/cpc_a_HWD_0.2_pv_north_hw_0.4_lag7_tm.pdf')

					fig = plt.figure(figsize =(6,5))
					plt.rcParams["font.family"] = "serif"
					ax, _, _ = ecdf(hw_intensity_blo_related, x_label=None)
					ax, _, _ = ecdf(hw_intensity_nonblo_related, y_label=None)
					ax.tick_params(labelsize = 13)
					plt.title('(g) HWI [TM index]',fontsize = 13)
					plt.ylabel('ECDF',fontsize = 13)
					plt.legend(['BRH', 'BURH'],prop={'size': 13})
					plt.show()
					fig.savefig(file_dir+ '/paper_figure/cpc_a_HWI_0.2_pv_north_hw_0.4_lag7_tm.pdf')



					# plot_cdf(hw_area_blo_related,hw_area_nonblo_related,nbins,nbins,"ECDF of heatwave area under blocked/nonblocked days", "Area",file_dir + '/heatwave_event/ECDF of heatwave area under blocked_nonblocked days_' + str(len_thres) + '_' + region[reg]+ '.png')
					# plot_cdf(hw_length_blo_related,hw_length_nonblo_related,nbins,nbins,"ECDF of heatwave length under blocked/nonblocked days", "length",file_dir + '/heatwave_event/ECDF of heatwave length under blocked_nonblocked days_' + str(len_thres) +'_' + region[reg]+ '.png')
					# plot_cdf(hw_extent_blo_related,hw_extent_nonblo_related,nbins,nbins,"ECDF of heatwave extent under blocked/nonblocked days", "extent",file_dir + '/heatwave_event/ECDF of heatwave extent under blocked_nonblocked days_' + str(len_thres) + '_' + region[reg]+ '.png')
					# plot_cdf(hw_intensity_blo_related,hw_intensity_nonblo_related,nbins,nbins,"ECDF of heatwave intensity under blocked/nonblocked days", "intensity",file_dir + '/heatwave_event/ECDF of heatwave intensity under blocked_nonblocked days_' + str(len_thres) + '_' + region[reg]+ '.png')



					# sio.savemat(file_dir + '/heatwave_event/kstest2_ori_of_hw_feature_len_' + str(len_thres) + '_extent_thres_'+ str(extent_thres) + '_' + blo_type +  '_' + hw_type + '_' + area_thres + detrend_type + '_' +region[reg] +'.mat',{'len_all':hw_length,'len_blo':hw_length_blo_related,'len_non':hw_length_nonblo_related})
					# sio.savemat(file_dir + '/heatwave_event/kstest2_ori_of_hw_feature_extent_ '+ str(len_thres) + '_extent_thres_'+ str(extent_thres) + '_' + blo_type +  '_' + hw_type + '_' + area_thres + detrend_type + '_' +region[reg] +'.mat',{'extent_all':hw_extent,'extent_blo':hw_extent_blo_related,'extent_non':hw_extent_nonblo_related})
					# sio.savemat(file_dir + '/heatwave_event/kstest2_ori_of_hw_feature_area_ '+ str(len_thres) + '_extent_thres_'+ str(extent_thres) + '_' + blo_type +  '_' + hw_type + '_' + area_thres + detrend_type + '_' +region[reg] +'.mat' ,{'area_all':hw_area,'area_blo':hw_area_blo_related,'area_non':hw_area_nonblo_related})
					# sio.savemat(file_dir + '/heatwave_event/kstest2_ori_of_hw_feature_intensity_ ' + str(len_thres) + '_extent_thres_'+ str(extent_thres) + '_' + blo_type +  '_' + hw_type + '_' + area_thres + detrend_type + '_'+ region[reg] +'.mat',{'intensity_all':hw_intensity,'intensity_blo':hw_intensity_blo_related,'intensity_non':hw_intensity_nonblo_related})
				# # pdb.set_trace()
		elif ksresult:
			if sub_region:
				region = ['high','mid','low']
				for reg in range(len(region)):
					if DEOR:
						idx_blo, = np.where((hw_blo_length_ratio > len_thres) & (hw_blo_extent_ratio > extent_thres) & (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))
						idx_non_blo, = np.where(((hw_blo_length_ratio <= len_thres) | (hw_blo_extent_ratio < extent_thres))& (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))
					if volume:
						idx_blo, = np.where((hw_blo_area_ratio > thres[i]) & (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))
						idx_non_blo, = np.where((hw_blo_area_ratio <= thres[i]) & (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))

					if deor_mean:
						idx_blo, = np.where(((hw_blo_length_ratio+hw_blo_extent_ratio)/2 > thres[i]) & (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))
						idx_non_blo, = np.where(((hw_blo_length_ratio+hw_blo_extent_ratio)/2 <= thres[i]) & (hw_center >= region_row[reg][0])  &  (hw_center < region_row[reg][1]))

					idx_num = idx_blo.shape[0]
					idx_non_num = idx_non_blo.shape[0]
					# pdb.set_trace()

					hw_area_blo_related = hw_area[idx_blo]
					hw_length_blo_related = hw_length[idx_blo]
					hw_extent_blo_related = hw_extent[idx_blo]

					hw_area_nonblo_related = hw_area[idx_non_blo]
					hw_length_nonblo_related = hw_length[idx_non_blo]
					hw_extent_nonblo_related = hw_extent[idx_non_blo]

					hw_intensity_blo_related = hw_intensity[idx_blo]
					hw_intensity_nonblo_related = hw_intensity[idx_non_blo]

					hw_length_75th_blo_related = hw_length_75th[idx_blo]
					hw_length_75th_nonblo_related = hw_length_75th[idx_non_blo]
					hw_extent_75th_blo_related = hw_extent_75th[idx_blo]
					hw_extent_75th_nonblo_related = hw_extent_75th[idx_non_blo]
					# pdb.set_trace()
					# area_ks[i,reg,:2] = stats.ks_2samp(hw_area_blo_related,hw_area_nonblo_related) # 2-sides
					# length_ks[i,reg,:2] = stats.ks_2samp(hw_length_blo_related,hw_length_nonblo_related)
					# extent_ks[i,reg,:2] = stats.ks_2samp(hw_extent_blo_related,hw_extent_nonblo_related)
					# intensity_ks[i,reg,:2] = stats.ks_2samp(hw_intensity_blo_related,hw_intensity_nonblo_related)

					area_ks[i,reg,:2] = kstest2(hw_area_blo_related,hw_area_nonblo_related,alpha = 0.05,tail = 'smaller')[1:] # one side
					# pdb.set_trace()
					length_ks[i,reg,:2] = kstest2(hw_length_blo_related,hw_length_nonblo_related,alpha = 0.05,tail = 'smaller')[1:]
					extent_ks[i,reg,:2] = kstest2(hw_extent_blo_related,hw_extent_nonblo_related,alpha = 0.05,tail =  'smaller')[1:]
					intensity_ks[i,reg,:2] = kstest2(hw_intensity_blo_related,hw_intensity_nonblo_related,alpha = 0.05,tail = 'smaller')[1:]


					area_ks[i,reg,2:] = idx_num,idx_non_num
					length_ks[i,reg,2:] = idx_num,idx_non_num
					extent_ks[i,reg,2:] = idx_num,idx_non_num
					intensity_ks[i,reg,2:] = idx_num,idx_non_num
					nbins = 100
					plot_cdf(hw_area_blo_related,hw_area_nonblo_related,nbins,nbins,"ECDF of heatwave area under blocked/nonblocked days", "Area",file_dir + '/paper_figure/cpc_ECDF of heatwave area under blocked_nonblocked days_' + str(len_thres) + '_' + region[reg]+ '.pdf')
					plot_cdf(hw_length_blo_related,hw_length_nonblo_related,nbins,nbins,"ECDF of heatwave length under blocked/nonblocked days", "length",file_dir + '/paper_figure/cpc_ECDF of heatwave length under blocked_nonblocked days_' + str(len_thres) +'_' + region[reg]+ '.pdf')
					plot_cdf(hw_extent_blo_related,hw_extent_nonblo_related,nbins,nbins,"ECDF of heatwave extent under blocked/nonblocked days", "extent",file_dir + '/paper_figure/cpc_ECDF of heatwave extent under blocked_nonblocked days_' + str(len_thres) + '_' + region[reg]+ '.pdf')
					plot_cdf(hw_intensity_blo_related,hw_intensity_nonblo_related,nbins,nbins,"ECDF of heatwave intensity under blocked/nonblocked days", "intensity",file_dir + '/paper_figure/cpc_ECDF of heatwave intensity under blocked_nonblocked days_' + str(len_thres) + '_' + region[reg]+ '.pdf')



		else:
			idx_blo, = np.where((hw_blo_length_ratio > len_thres) & (hw_blo_extent_ratio > extent_thres))
			idx_num = idx_blo.shape[0]
			idx_non_blo, = np.where((hw_blo_length_ratio <= len_thres) | (hw_blo_extent_ratio < extent_thres))
			idx_non_num = idx_non_blo.shape[0]

			hw_area_blo_related = hw_area[idx_blo]
			hw_length_blo_related = hw_length[idx_blo]
			hw_extent_blo_related = hw_extent[idx_blo]

			hw_area_nonblo_related = hw_area[idx_non_blo]
			hw_length_nonblo_related = hw_length[idx_non_blo]
			hw_extent_nonblo_related = hw_extent[idx_non_blo]

			hw_intensity_blo_related = hw_intensity[idx_blo]
			hw_intensity_nonblo_related = hw_intensity[idx_non_blo]

			hw_length_75th_blo_related = hw_length_75th[idx_blo]
			hw_length_75th_nonblo_related = hw_length_75th[idx_non_blo]
			hw_extent_75th_blo_related = hw_extent_75th[idx_blo]
			hw_extent_75th_nonblo_related = hw_extent_75th[idx_non_blo]
			# pdb.set_trace()


			sio.savemat(file_dir + '/heatwave_event/ori_of_hw_feature_len_' + str(len_thres) + '_extent_thres_'+ str(extent_thres) + '_' + blo_type +  '_' + hw_type + '_' + area_thres + detrend_type +'.mat',{'len_all':hw_length,'len_blo':hw_length_blo_related,'len_non':hw_length_nonblo_related})
			sio.savemat(file_dir + '/heatwave_event/ori_of_hw_feature_extent_ '+ str(len_thres) + '_extent_thres_'+ str(extent_thres) + '_' + blo_type +  '_' + hw_type + '_' + area_thres + detrend_type +'.mat',{'extent_all':hw_extent,'extent_blo':hw_extent_blo_related,'extent_non':hw_extent_nonblo_related})
			sio.savemat(file_dir + '/heatwave_event/ori_of_hw_feature_area_ '+ str(len_thres) + '_extent_thres_'+ str(extent_thres) + '_' + blo_type +  '_' + hw_type + '_' + area_thres + detrend_type +'.mat' ,{'area_all':hw_area,'area_blo':hw_area_blo_related,'area_non':hw_area_nonblo_related})
			sio.savemat(file_dir + '/heatwave_event/ori_of_hw_feature_intensity_ ' + str(len_thres) + '_extent_thres_'+ str(extent_thres) + '_' + blo_type +  '_' + hw_type + '_' + area_thres + detrend_type +'.mat',{'intensity_all':hw_intensity,'intensity_blo':hw_intensity_blo_related,'intensity_non':hw_intensity_nonblo_related})
	

	# sio.savemat(file_dir + '/heatwave_event/kstest_of_hw_blo_related_non_related_feature_result_'+ blo_type +'_' + hw_type + '.mat',{'area':area_ks,'length':length_ks,'extent':extent_ks,'intensity':intensity_ks})
		# pdb.set_trace()
	''' initial submission '''
	# sio.savemat(file_dir + '/heatwave_event/kstest2_smaller_of_deor_mean_hw_blo_related_non_related_feature_result_'+ blo_type +'_' + hw_type + '_lag7.mat',{'area':area_ks,'length':length_ks,'extent':extent_ks,'intensity':intensity_ks})
	''' first revision '''
	np.savez(file_dir + '/cpc_tmp/kstest2_smaller_of_deor_mean_hw_blo_related_non_related_feature_result_'+ blo_type +'_' + hw_type + '_lag7.mat', area = area_ks, length = length_ks, extent = extent_ks, intensity = intensity_ks)
	pdb.set_trace()


	# pdb.set_trace()
def main_hw_fea_trend():
	ratio= True
	file_dir = '/home/user/Documents/research/project1'
	# fea = sio.loadmat(file_dir +'/heatwave_event/feature/all_hw_event_feature_ratio_0.5_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_land_only_labeled_2d_nondetrended.mat')['hw_blo_feature']
	
	''' initial submission'''
	# fea = sio.loadmat(file_dir +'/heatwave_event/feature/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_0.4_daily_land_only_labeled_2d_detrended_sig_only.mat')['hw_blo_feature']
	
	''' first revision '''
	# fea = np.load(file_dir +'/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_0.4_daily_land_only_labeled_2d_nondetrended.npy')
	# fea = np.load(file_dir +'/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_pv1.2_0.4_daily_land_only_labeled_2d_detrended.npy')
	fea = np.load(file_dir + '/cpc_tmp/all_hw_event_feature_ratio_0.4_area_3_duration_3_extend_0_35_75N_with_label_tm90_nospatial_filter_daily_land_only_labeled_2d_nondetrended_new.npy')

	
	idx_0= np.where(fea[:,0]==0)
	fea = np.delete(fea,idx_0,axis=0)
	fea_df =  pd.DataFrame(fea, columns = ['year','event_num','area','length','extent','co_area','co_length','co_extent','co_area_ratio','co_length_ratio','co_extent_ratio','center_row','center_col','hw_label','hw_intensity','hw_length_75th','hw_extent_75th'])

	pdb.set_trace()
	hw_length = fea_df.length
	hw_extent = fea_df.extent
	hw_intensity = fea_df.hw_intensity
	hw_area = fea_df.area
	hw_year = fea_df.year

	len_thres = [0,3,4,5,6]
	extent_thres = [0,30,100,300,1000]
	area_thres = [0,100,300,1000,3000]
	# intensity_thres = [0,100,300,500,3000]
	intensity_thres = [0,100,300,1000,3000]

	len_trend = np.zeros([39,5],dtype = np.float32)
	extent_trend = np.zeros([39,5],dtype = np.float32)
	area_trend = np.zeros([39,5],dtype = np.float32)
	intensity_trend = np.zeros([39,5],dtype = np.float32)


	for year in range(1979,2018):
		for i in range(5):
			# pdb.set_trace()
			len_trend[year-1979,i] = np.where((hw_year.values==year) & (hw_length.values > len_thres[i]))[0].shape[0]
			area_trend[year-1979,i] = np.where((hw_year.values==year) & (hw_area.values > area_thres[i]))[0].shape[0]
			intensity_trend[year-1979,i] = np.where((hw_year.values==year) & (hw_intensity.values > intensity_thres[i]))[0].shape[0]
			extent_trend[year-1979,i] = np.where((hw_year.values==year) & (hw_extent.values > extent_thres[i]))[0].shape[0]

	if ratio:
		for col in range(5):
			len_trend[:,col] = len_trend[:,col] / np.mean(len_trend,axis = 0)[col]
			area_trend[:,col] = area_trend[:,col] / np.mean(area_trend,axis = 0)[col]
			extent_trend[:,col] = extent_trend[:,col] / np.mean(extent_trend,axis = 0)[col]
			intensity_trend[:,col] = intensity_trend[:,col] / np.mean(intensity_trend,axis = 0)[col]

	len_cor = cor_relation_1d(len_trend)
	area_cor = cor_relation_1d(area_trend)
	extent_cor = cor_relation_1d(extent_trend)
	intensity_cor = cor_relation_1d(intensity_trend)
	pdb.set_trace()

	''' initial submission '''
	# sio.savemat(file_dir + '/heatwave_event/hw_fea_trend_ratio_new_pv_1.2_0.4_hw_0.4.mat',{'len':len_trend,'area':area_trend, 'intes':intensity_trend,'extent':extent_trend})

	''' first revision '''
	np.savez(file_dir + '/cpc_tmp/hw_fea_trend_ratio_tm90_hw_0.4_nondetrend_new.npz',len = len_trend, area = area_trend, intes = intensity_trend, extent = extent_trend)
	# sio.savemat(file_dir + '/heatwave_event/hw_fea_trend_cor.mat',{'len':len_cor,'area':area_cor, 'intes':intensity_cor,'extent':extent_cor})


def main_change_point():
	change_point = 18
	perst_day  = 5 # day 3 or 5
	long_perst = False
	prob = False
	hw_sum_3d, blo_sum_3d, hw_sum, blo_sum = load_data()
	
	if long_perst:
		hw_sum = time_filter(blo_sum_3d,perst_day) 
		hw_sum = np.transpose(hw_sum,[2,0,1])
		hw_sum = np.reshape(hw_sum,[92,39,110,140],order='F')
		blo_sum = hw_sum

	hw_minus= block_freq(hw_sum[:,18:,:,:],dimen = 'grid')/(hw_sum.shape[1]-change_point)- block_freq(hw_sum[:,:18,:,:],dimen = 'grid')/change_point
	hw_part = np.where(hw_minus>=1.2)[0].shape[0]/np.where(hw_minus!=0)[0].shape[0]

	print(hw_part)
	pdb.set_trace()
	
	blo_freq_be = block_freq(blo_sum[:,:18,:,:],dimen = 'grid')/change_point
	# hw_freq_be[np.where(hw_freq_be ==0)] = -1
	blo_freq_af = block_freq(blo_sum[:,18:,:,:],dimen = 'grid')/(hw_sum.shape[1]-change_point)
	blo_minus =  blo_freq_af - blo_freq_be
	# hw_freq_af[np.where(hw_freq_af ==0)] = -1

	if prob:
		blo_and_hw = co_occur(hw_sum,blo_sum)
		blo_in_hw_minus = condi_prob (blo_and_hw[:,18:,:,:], hw_sum[:,18:,:,:], dimen='grid') - condi_prob (blo_and_hw[:,:18,:,:], hw_sum[:,:18,:,:], dimen='grid')
		blo_in_hw_be = condi_prob (blo_and_hw[:,:18,:,:], hw_sum[:,:18,:,:], dimen='grid')
		blo_in_hw_af = condi_prob (blo_and_hw[:,18:,:,:], hw_sum[:,18:,:,:], dimen='grid')

	# clevs = np.arange(-1,6,1)
	clevs = np.arange(-8,6,2)
	out_title = 'change_blo_freq_minus_day_'+ str(perst_day)
	outfig = 'change_ blo_freq_minus_day_'+ str(perst_day) +'.png' 
	geomap_china(blo_minus,clevs,out_title,outfig)

	# out_title = 'change_1996_condi_proba_day_'+ str(perst_day)
	# outfig = 'change_1996_condi_proba_day_'+ str(perst_day) +'.png' 
	# geomap_china(blo_in_hw_minus,clevs,out_title,outfig)


	# out_title = 'change_1996_before_blo_freq_day_'+ str(perst_day)
	# outfig = 'change_1996_before_blo_freq_day_'+ str(perst_day) +'.png'
	# geomap_china(hw_freq_be,clevs,out_title,outfig) 

	# out_title = 'change_1996_after_blo_freq_day_'+ str(perst_day)
	# outfig = 'change_1996_after_pro_blo_freq_day_'+ str(perst_day) +'.png' 
	# geomap_china(hw_freq_af,clevs,out_title,outfig)



def event_extent(labeled_hw):
	yearly_extent = np.zeros([labeled_hw.shape[1],200],dtype = np.int32) # maxinum spatial extent
	yearly_length = np.zeros([labeled_hw.shape[1],200],dtype =  np.int16) # maxium temporal length
	for year in range(labeled_hw.shape[1]):
		labeled_sample = labeled_hw[:,year,:,:]
		label_max = np.max(labeled_sample)
		for idx in range(1,int(label_max)+1):
			labeled_sample_copy = labeled_sample.copy()
			labeled_sample_copy[np.where(labeled_sample_copy!=idx)] = 0
			labeled_sum = np.sum(labeled_sample_copy,axis =0)/idx
			# pdb.set_trace()
			yearly_extent[year,idx] = np.array(np.where(labeled_sum>0))[0].shape[0]
			yearly_length[year,idx] = labeled_sum.max()

	return yearly_extent,yearly_length

def event_thres(data,thres_list,ratio,cate):
	data_all_year = np.zeros([data.shape[0],len(thres_list)],dtype = np.float32)
	for idx in range(len(thres_list)):
		thres = thres_list[idx]
		data_copy = data.copy()
		data_copy[np.where(data_copy<=thres)] = 0
		data_copy[np.where(data_copy>thres)] = 1
		data_year = np.sum(data_copy, axis = 1) # event
		if ratio:
			data_year = data_year/np.mean(data_year)
		data_all_year[:,idx] = data_year
		print("data_year:",data_year)
		slope,intercept, r, p, std_err=linregress(range(39),np.squeeze(data_year))
		print ("slope= ",slope)
		print ("pvalue= ",p)

	if cate:
		for idx in range(len(thres_list)-1):
			data_all_year[:,idx] = data_all_year[:,idx]- data_all_year[:,idx+1]
			print("data_year_cate:",data_year)
			slope,intercept, r, p, std_err=linregress(range(39),np.squeeze(data_all_year[:,idx]))
			print ("cate_slope= ",slope)
			print ("cate_pvalue= ",p)

	return data_all_year 

def main_event_feature(hw_labeled):
	''' find the spatial centroid of each event'''
	''' event feature: year, area centroid,extent,length'''
	data = hw_labeled
	# data = sio.loadmat('./north_asia/heatwave_labeled_duration3_measure_label_valid_north_asia.mat')['hw_sum_labeled']
	data = data.astype(np.int16)
	event_all = np.zeros([1,8],dtype = np.float16)
	event_fea = np.zeros([1,8],dtype = np.float16)
	area_all = []
	for year in range(data.shape[1]):
		sample_year = data[:,year,:,:]
		props = measure.regionprops(sample_year)
		area = [ele.area for ele in props]
		print(area)
		area_all.append(area)
		for idx in range(1,np.nanmax(sample_year)+1):
			# print(idx)
			# event_fea.append(year+1979)
			event_fea[0,0]=year+1979
			event_sample = sample_year.copy()
			event_area0 = np.where(event_sample == idx)[0].shape[0]
			event_sample[np.where(event_sample!= idx)]=0
			event_area = np.where(event_sample > 0)[0].shape[0] 
			# event_fea.append(event_area)
			event_fea[0,1]= event_area
			# print(event_area)
			
			event_sample_sum = np.sum(event_sample, axis = 0)/idx
			event_centroid = np.array(ndimage.measurements.center_of_mass(event_sample_sum))
			event_centroid = np.around(event_centroid,decimals =2)
			# event_fea.append(event_centroid)
			event_fea[0,2:4]=event_centroid
			
			event_extent = np.array(np.where(event_sample_sum>0))[0].shape[0]
			event_length = event_sample_sum.max()
			# event_fea.append(event_extent)
			event_fea[0,4]= event_extent
			# event_fea.append(event_length)
			event_fea[0,5]= event_length
			event_fea[0,6] = idx
			event_fea[0,7] = event_area0
			event_all = np.append(event_all,event_fea,axis = 0)
			print(event_all)
			# pdb.set_trace()
	# pdb.set_trace()
	event_all = np.delete(event_all, np.where(np.isnan(event_all)), axis=0)
	print(event_all)
	# pdb.set_trace()
	# sio.savemat('./north_asia/heatwave_event_feature_duration3_idx_valid_north_asia.mat',{'event_feature':event_all})

	area_all_new = []
	for idx in range(len(area_all)):
		for idx2 in range(len(area_all[idx])):
			area_all_new.append(area_all[idx][idx2])
	print(len(area_all_new))
	area_all_new = np.array(area_all_new)
	print(area_all_new.max())
	print(area_all_new.min())
	# sio.savemat('heatwave_event_feature_area_all_duration3.mat',{'event_feature':np.array(area_all_new)})

	# pdb.set_trace()
	return event_all


def main_event_feature_trend():
	event_feature = sio.loadmat('./north_china/heatwave_event_feature_duration3_idx_valid_north.mat')['event_feature']
	year_length_median = np.zeros([39,1],dtype = np.float16)
	for year in range(1979,2018):
		idx = np.where(event_feature[:,0]==year)
		# pdb.set_trace()
		year_length = event_feature[idx,-4]
		# year_length_median[year-1979,0] = np.nanmean(year_length)
		year_length_median[year-1979,0] = np.nanmean(year_length)
		# pdb.set_trace()
	cor_kp = cor_relation_1d(year_length_median)
	pdb.set_trace()



def main_menclo_test():
	# ********** Mento calor test *************** 
	plot_only = True
	random_repeat = 1000
	withta = False
	sign = True
	dif = False # difference
	factor = False # ratio
	day_thres = 3
	long_duration = False
	north = False
	anom_hw = False # p(hw|blocking)/(hw|summer) 
	whole_summer = True
	diff_ratio = True

	# **************** load data ***************************
	file_dir = '/home/user/Documents/research/project1'
	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	ta_90th_3d = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N_detrend_105_39_sig_only.mat')['hw']
	ta_90th = np.transpose(ta_90th_3d,[2,0,1])
	ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')

	pv_list = [1.3,1.2]
	ratio_list = [0.35,0.5]

	# pv_list = [1.1]
	# ratio_list = [0.45]

	if not plot_only:
		for pv in pv_list:
			for ratio in ratio_list:
				blo_sum_weak_3d = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv'+str(pv)+'_5day_2dtrack_weighted_3d_35_75N_all_ratio_' + str(ratio)+'_daily_extent_100.mat')['blocking']
				blo_sum_weak = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv'+ str(pv)+'_5day_2dtrack_weighted_4d_35_75N_all_ratio_'+ str(ratio)+'_daily_extent_100.mat')['blocking']

				blo_type = '_pv_' + str(pv) + '_'+ str(ratio) +'_daily_'
				# blo_type = '_tm90_with_spatialfilter_'
				hw_type = '_tamax_90th_detrend_'
				ta_type = '_'

				hw_sum = ta_90th
				hw_sum_3d = ta_90th_3d
				
				blo_sum = blo_sum_weak
				blo_sum_3d = blo_sum_weak_3d
				hw_sum_3d[np.isnan(hw_sum_3d)] = 0 
				hw_sum[np.isnan(hw_sum)] = 0 


		# for day in range(blo_sum.shape[0]):
		# 	for year in range(blo_sum.shape[1]):
		# 		blo_sum[day,year,:,:][np.where(land_mask == 0 )] = 0
		# 		hw_sum[day,year,:,:][np.where(land_mask == 0 )] = 0

		# 	for day in range(blo_sum.shape[0]):
		# 	for year in range(blo_sum.shape[1]):
		# 		blo_sum[day,year,:,:][np.where(land_mask == 0 )] = 0
		# 		hw_sum[day,year,:,:][np.where(land_mask == 0 )] = 0

				if withta:
					tamax_75th = sio.loadmat('./ta-max/tamax_dif_75th_summer_15win.mat')['ta_75th'] 
					tamax_75th_4d = np.transpose(tamax_75th,[2,0,1])
					tamax_75th_4d = np.reshape(tamax_75th_4d,[92,39,80,180],order = 'F')


					blo_and_ta = co_occur(tamax_75th_4d,blo_sum,False,False)
					hw_and_blo_and_ta = co_occur(hw_sum,blo_and_ta,False,False)
					hw_in_blo_ta =  condi_prob(hw_and_blo_and_ta,blo_and_ta,dimen = 'grid')


					hw_and_ta = co_occur(tamax_75th_4d,hw_sum,False,False)
					hw_in_ta = condi_prob(hw_and_ta,tamax_75th_4d,dimen = 'grid')
					hw_in_ta [np.where(hw_in_ta==1)] = 0
					
					blo_sum = blo_and_ta

				elif factor:
					if anom_hw:
						# hw_freq_anom = np.divide(hw_in_blo, hw_freq_sum) # p(hw/blo)/p(summer)
						hw_freq_anom = np.divide(hw_in_blo_ta, hw_in_ta)
						# pdb.set_trace()
					else:
						# blo_freq_anom = np.divide(blo_in_hw, blo_freq_sum)
						blo_freq_anom = blo_in_hw
				elif dif:
					if anom_hw:
						hw_freq_anom = hw_in_blo - hw_freq_sum
					else:
						blo_freq_anom = blo_in_hw - blo_freq_sum
				else:
					hw_and_blo = co_occur(hw_sum,blo_sum,False,False)
					hw_in_blo =  condi_prob(hw_and_blo,blo_sum,dimen = 'grid')
					# sio.savemat(file_dir + '/temporal_result/hw_in_blo_' + blo_type + '-' +hw_type +'sig_test.mat',{'hw_in_blo':hw_in_blo})
					# pdb.set_trace()
					print(np.nanmax(hw_in_blo))
					blo_in_hw = condi_prob(hw_and_blo,hw_sum,dimen = 'grid')
					# hw_freq_anom = blo_in_hw
					# pdb.set_trace()
					hw_freq_anom = hw_in_blo

				# ************ p(hw/blo) *************** 
				# if long_duration:
				# 	hw_sum = time_filter_new(hw_sum,day_thres)
				# 	blo_sum = time_filter_new(blo_sum,day_thres)

				# hw_blo = co_occur(hw_sum, blo_sum,False,False)

				# if anom_hw:
				# 	hw_in_blo = condi_prob(hw_blo, blo_sum,dimen = 'grid')
				# 	hw_freq_sum = block_freq (hw_sum,dimen = 'grid')/92/39
				# 	sio.savemat('./north_china/hw_in_blo_sum.mat',{'hw_in_blo':hw_in_blo})
				# 	sio.savemat('./north_china/hw_freq_sum.mat',{'hw_freq':hw_freq_sum})
				# else:
				# 	blo_in_hw = condi_prob(hw_blo, hw_sum,dimen = 'grid')
				# 	blo_freq_sum = block_freq (blo_sum,dimen = 'grid')/92/39
				# 	sio.savemat('./north_china/blo_in_hw_sum.mat',{'blo_in_hw':blo_in_hw})
				# 	sio.savemat('./north_china/blo_freq_sum.mat',{'blo_freq':blo_freq_sum})
				# pdb.set_trace()

				# pdb.set_trace()


			 	## test significance ## 

				if sign:
					for row in range(blo_sum.shape[2]):
						print(row)
						for col in range(blo_sum.shape[3]):
							perst_ori = []
							if anom_hw:
								n_days = np.nansum(blo_sum[:,:,row,col]) # hw_anom
							else:
								n_days = np.nansum(hw_sum[:,:,row,col])
							# print("n_days=",n_days)
							# print("n_days/92/39=", n_days/92/39)
							prob = -1*np.ones([random_repeat,1],dtype = np.float16)
							if whole_summer:
								for year in range(blo_sum.shape[1]):
									if anom_hw:
										idx0, = np.where(blo_sum[:,year,row,col]==0) # hw_anom
									else:
										idx0, =np.where(hw_sum[:,year,row,col]==0) 
										# print(idx0)
									if idx0[-1] < 91:
										idx0 = np.append(idx0,92)
									idx1= np.append(idx0,0)
									idx0= np.insert(idx0,1,0)
									perst = idx1-idx0-1
									idx = np.where(perst<1) # only for blo or hw only
									perst = np.delete(perst,idx)
									perst = [per for per in perst]
									if len(perst) > 0:
										perst_ori.append(perst)
								# print("perst_ori", perst_ori)

								new_perst_ori = []
								for sublist in perst_ori:
									for item in sublist:
										new_perst_ori.append(item)
								print("perst_sum=",np.sum(a for a in new_perst_ori))

								for time in range(random_repeat):
									rd_day = []
									# for days in perst_ori:
									# 	idx_rd = np.random.randint(0,92*39-days[0])
									# 	add_day = [day for day in range(idx_rd,idx_rd+days[0])]  # multi blocking in a year if day[0] only consider the first blocking in a year
									# 	rd_day.append(add_day) # unique or not 
									# print("rd_day:",rd_day)

									for days in new_perst_ori:
										idx_rd = np.random.randint(0,92*39-days)
										add_day = [day for day in range(idx_rd,idx_rd+days)]  # multi blocking in a year if day[0] only consider the first blocking in a year
										rd_day.append(add_day) # unique or not 
										# pdb.set_trace()

									new_rd_day = []
									for sublist in rd_day:
										for item in sublist:
											new_rd_day.append(item)

									if anom_hw:
										hw_rd = hw_sum_3d[row,col,new_rd_day]
										hw_rd_freq = np.sum(hw_rd)/len(new_rd_day) # hw_frequency in random day # for ta_75 -- hw_freq in tamax_75
										print("random_days=", len(new_rd_day))
										prob[time] = hw_rd_freq

									else:
										blo_rd = blo_sum_3d[row,col,new_rd_day]
										blo_rd_freq = np.sum(blo_rd)/len(new_rd_day) # blo_frequency in random day
										prob[time] = blo_rd_freq

							else:
								for time in range(random_repeat):
									idx1, = np.where(tamax_75th[row,col,:]==1)
									idx_rd = np.random.randint(0,idx1.shape[0],size = int(n_days))
									new_rd_day = idx1[idx_rd]

									# print("new_rd_day",new_rd_day)
									if anom_hw:
										hw_rd = hw_sum_3d[row,col,new_rd_day]
										hw_rd_freq = np.sum(hw_rd)/len(new_rd_day) # hw_frequency in random day # for ta_75 -- hw_freq in tamax_75
										# print("random_days=", len(new_rd_day))
										prob[time] = hw_rd_freq

									else:
										blo_rd = blo_sum_3d[row,col,new_rd_day]
										blo_rd_freq = np.sum(blo_rd)/len(new_rd_day) # blo_frequency in random day
										prob[time] = blo_rd_freq

							prob[np.where(prob==-1)]=np.nan
							prob = prob[~np.isnan(prob)]
							if len(prob) > 0:
								pro_95 = np.percentile(prob,95)
								pro_5 = np.percentile(prob,95)
								# pdb.set_trace()

								if anom_hw:
									if hw_in_blo[row,col] > pro_95 :
									# if hw_in_blo_ta[row,col] > pro_95 :
										hw_freq_anom[row,col] = hw_freq_anom[row,col]
									else:
										hw_freq_anom[row,col] = np.nan
								else:
									if blo_in_hw[row,col] > pro_95 :
										hw_freq_anom[row,col] = hw_freq_anom[row,col]
									else:
										hw_freq_anom[row,col] = np.nan
							else:
								if anom_hw:
									hw_freq_anom[row,col] = np.nan
								else:
									# blo_freq_anom[row,col] = np.nan
									hw_freq_anom[row,col] = np.nan

				sio.savemat(file_dir + '/temporal_result/significant_mentocor_test_P(heatwave|blocking)' +  blo_type + hw_type + ta_type + str(random_repeat) +'.mat',{'anom':hw_freq_anom})
				# sio.savemat(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)' +  blo_type + hw_type + ta_type + str(random_repeat) +'.mat',{'anom':hw_freq_anom})

				# ******************************* plot figure ****************************************
				# # clevs = np.arange(0,5.5,0.5)
				# # clevs = np.array([1/3,1/2.5,1/2,1/1.5,1,1.5,2,2.5,3])
				
				# # clevs = np.arange(0,0.525,0.025)
				# clevs = np.arange(0,0.55,0.05)
				# # clevs = np.arange(0,0.75,0.05)

				# # hw_freq_anom = sio.loadmat(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)_blo_1.0_0.35__tamax_90th__100.mat')['anom']
				# hw_freq_anom[np.where(land_mask==0)] = np.nan
				# # clevs = np.arange(0,0.75,0.05)
				# # clevs = np.array([1/5,1/4,1/3,1/2,1,2,3,4,5])
				# # out_title = 'anom_diff_sig_p_blo_in_hw__p_blo_sum_100_duration_5repeat'
				# # outfig = 'anom_diff_sig_p_blo_in_hw_p_blo_sum_100_repeat_duration_5.png'

				# # out_title = 'ratio of anomaly in P(blocking|heatwave)'
				# # outfig = './era-pv/anom_ratio_sig_hw_in_blo_100_repeat_duration_3_tm90.png'

				# # out_title = 'P(blocking|heatwave) significance tested by monto carlo'
				# # outfig = './era-pv/north_asia/P(blocking|heatwave) significance tested by monto carlo_pvweak_0.7_98.png'

				# # out_title = 'ratio of anomaly in P(heatwave | blocking)'
				# # outfig = './era-pv/anom_ratio_sig_p_blo_in_hw_p_blo_sum_100_repeat_hw_5_blo_weak_valid.png'

				# # out_title = 'try'
				# # outfig = 'try.png'

				# out_title = 'P(heatwave|blocking)'
				# outfig = file_dir + '/result_figure/significant_mentocor_test_P(heatwave|blocking)_proj' + blo_type + hw_type + ta_type + str(random_repeat) +'.png'

				# # out_title = 'P(blocking|heatwave)'
				# # outfig = file_dir + '/result_figure/significant_mentocor_test_P(blocking|heatwave)_proj' + blo_type + hw_type + ta_type + str(random_repeat) +'.png'
				
				# single_data = True

				# geomap_china(hw_freq_anom,hw_freq_anom, clevs,clevs,out_title,outfig, single_data)
				# # geomap_china(blo_freq_anom,blo_freq_anom, clevs,clevs,out_title,outfig, single_data)

	if plot_only:
		if diff_ratio:
			blo_in_hw_pv = sio.loadmat(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)_pv_1.2_0.4_daily__tamax_90th_detrend__1000.mat')['anom']
			hw_in_blo_pv = sio.loadmat(file_dir + '/temporal_result/significant_mentocor_test_P(heatwave|blocking)_pv_1.2_0.4_daily__tamax_90th_detrend__1000.mat')['anom']

			blo_in_hw_tm = sio.loadmat(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)_tm90_no_spatialfilter__tamax_90th_detrend__1000.mat')['anom']
			hw_in_blo_tm = sio.loadmat(file_dir + '/temporal_result/significant_mentocor_test_P(heatwave|blocking)_tm90_no_spatialfilter__tamax_90th_detrend__1000.mat')['anom']
			
			hw_freq_sum = block_freq (hw_sum,dimen = 'grid')/92/39
			blo_freq_sum_pv = block_freq (blo_sum_weak,dimen = 'grid')/92/39
			blo_freq_sum_tm = block_freq (blo_sum_tm90,dimen = 'grid')/92/39
			
			hw_freq_dif_tm = hw_in_blo_tm - hw_freq_sum
			hw_freq_dif_pv = hw_in_blo_pv - hw_freq_sum

			blo_freq_dif_tm = blo_in_hw_tm - blo_freq_sum_tm
			blo_freq_dif_pv = blo_in_hw_pv - blo_freq_sum_pv

			# hw_freq_ratio_tm = np.divide(hw_in_blo_tm, hw_freq_sum)
			# hw_freq_ratio_pv = np.divide(hw_in_blo_pv, hw_freq_sum)

			blo_freq_ratio_tm = np.divide(blo_in_hw_tm, blo_freq_sum_tm)
			blo_freq_ratio_pv = np.divide(blo_in_hw_pv, blo_freq_sum_pv)
			blo_freq_ratio_tm[np.where(land_mask==0)] = np.nan
			blo_freq_ratio_pv[np.where(land_mask==0)] = np.nan
			# blo_freq_ratio_tm[np.where(blo_freq_ratio_tm>5)] = 5
			# blo_freq_ratio_pv[np.where(blo_freq_ratio_pv>5)] = 5
			print(np.nanmean(blo_freq_ratio_tm[0:40,])) 
			print(np.nanmean(blo_freq_ratio_pv[0:40,]))

			blo_in_hw_tm[41:60,:]=np.nan

			# idx = np.where(blo_in_hw_tm>0.25)
			# idx_nan = np.isnan(blo_in_hw_tm)
			# print(idx[0].shape[0]/(80*180-np.sum(idx_nan)))
			# pdb.set_trace()


			print(np.nanmean(blo_in_hw_tm))
			print(np.where(blo_in_hw_tm>0.5))
			pdb.set_trace()


			
			clevs1  = np.arange(0,0.55,0.05)
			outfig1 = file_dir + '/result_figure/conditional_prob_hw_freq_dif.png'
			title1 = "Difference between P(heatwave|blocking) and P(heatwave)"
			single_data = False
			# geomap_china(hw_freq_dif_tm, hw_freq_dif_pv, clevs1,clevs1,title1,title1,outfig1,single_data) # 4

			clevs1  = np.arange(0,0.55,0.05)
			outfig2 = file_dir + '/result_figure/conditional_prob_blo_freq_dif.png'
			title2 = "Difference between P(blocking|heatwave) and P(blocking)"
			single_data = False
			# geomap_china(blo_freq_dif_tm, blo_freq_dif_pv, clevs1,clevs1,title2,title2,outfig2,single_data) # 4

			clevs3  = np.arange(1,10.5,0.5)
			outfig3 = file_dir + '/result_figure/conditional_prob_ratio_hw_blo_land.png'
			title3 = "Multiples of P(heatwave|blocking) to P(heatwave)"
			# title3 = "Ratio of P(blocking|heatwave) to P(blocking)"
			single_data = False
			# geomap_china(hw_freq_ratio_tm, hw_freq_ratio_pv, clevs3,clevs3,title3,title3,outfig3,single_data) # 4
			geomap_china(blo_freq_ratio_tm, blo_freq_ratio_pv, clevs3,clevs3,title3,title3,outfig3,single_data)


		else:
			#path = sorted(glob.glob(file_dir + '/blocking/pv_mat/*pv_weak_1.2_5day_track_ori.mat'))
			path2 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P*_pv_1.2_0.4_daily__tamax_90th_detrend__1000.mat'))
			path1 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P*_tm90_no_spatialfilter__tamax_90th_detrend__1000.mat'))

			# path1 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P(heatwave|blocking)*pv*_*1000.mat'))
			# path2 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)*pv*_*1000.mat'))

			# path1 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P(heatwave|blocking)_*tm90*with*1000.mat'))
			# path2 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)_*tm90*with*1000.mat'))\



			for i in range(len(path1)):
				pdb.set_trace()
				# print(path1[i].find('P'))
				print(path1[i])
				# pdb.set_trace()
				# print(np.where(path1[i]=='v'))
				pv_type = path1[i][102:120]
				title1 = path1[i][81:101]
				title2 = path2[i][81:101]
				# pdb.set_trace()
				data1 = sio.loadmat(path1[i])['anom']
				data2 = sio.loadmat(path2[i])['anom']
				data1[np.where(land_mask==0)] = np.nan
				data2[np.where(land_mask==0)] = np.nan
				clevs = np.arange(0,0.55,0.05)
				# clevs = np.arange(0,0.55,0.05)
				# outfig = file_dir + '/result_figure/conditional_hw_in_blo.png'
				outfig = file_dir + '/result_figure/conditional_prob_' + pv_type + '_two_blocking_index.png'
				# title1 = 'p(heatwave|blocking)'
				# title2 = 'p(heatwave|blocking)'
				# title1 = 'p(blocking|heatwave)'
				# title2 = 'p(blocking|heatwave)'
				single_data = False
				geomap_china(data1, data2, clevs,clevs,title1,title2,outfig,single_data) # 4


if __name__ == "__main__":
	# main() #****** blocking frequency and hw in blo ********  # 1 the spatial distribution and the association 
	# main_temporal() # temporal trend....
	# main_daily_gridnum() # seasonality
	# main_hw_freq_trend() # spatial trend
	# main_cor_2d()  # **** spatial correlation of heatwave and ta
	# main_hw_fea_trend()
	# main_plot_line()
	# main_hw_freq_trend()
	# main_menclo_test()

	##  ****** event based *******
	# main_hw_event()
	# main_hw_blo_evnt()
	# main_hw_fea_trend()
	# main_hw_blo_event_lag()
	main_blo_related_hw()
	# main_top_event()

	# ************ OR ratio **************
	# OR_ratio()
	# main_lag()
	# main_lag_ev_based()
	# main_plot_pie()
	# main_blo_related_hw_ev()

	# main_p_hw_blo_ratio() # ******* significance test   # 2 significance of the association 
	# main_composite()
	# main_change_point()
	# main_hw_ta()
	# main_perst()

	# mon_blo_freq()
	# main_big_event()

	# main_hw_freq_trend()  # ***** spatial trend of hw or blo
	# main_event_feature_trend() # length or extent increased or not??


	# main_event_simple()
	# main_connected_event() # 1th
	# main_event_feature() # feature for each event 2th
	# main_event_cluster()
	# biseral_1d() # hw_blo biserial correlation
	# # time_step for overlapping top 10 event
	# main_boots_trap()
	# main_mass_point()
	

	# main_freq_prob()

	# main_concurrence()
