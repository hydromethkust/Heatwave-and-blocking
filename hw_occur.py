import numpy as np 
import scipy.io as sio
from PIL import Image
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import tifffile as tiff
from plot_code import geo_grid, geo_grid_2,plot_bar,geo_plot_point,plot_cdf,tsplot,plot_grid
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
# import pandas as pd

from hw_blo_basic import pv_overlap_check,pv_overlap_check_2d,pv_thres  # for the big event detection
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from hw_blo_basic import load_data,mon_blo_freq, make_ax, daynum, gridnum, daily_gridnum, persistence, block_freq, co_occur, condi_prob, cor_relation_2d, cor_relation_1d, geomap_china,lon_mean_blocking,lag_co_occur,event_number


def main_temporal():
	norm = False
	south_east = False
	north = False
	ratio = True
	trend = True
	correlation = False
	long_duration = False
	blo_related = False
	trend_of_prob = False
	norm_prob = False
	withta = False
	file_dir = '/home/user/Documents/research/project1'

	if trend:
		monthly = False
		hw_sum_3d, blo_fassum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
		# pdb.set_trace()
		# blo_sum = blo_sum_weak

		# blo_ta = sio.loadmat('./hw_blo_ta/blo_pv_1.0_and_ta_75th.mat')['blo_ta']
		# ta = sio.loadmat('./hw_blo_ta/tamax_dif_75th_summer_15win.mat')['ta_75th']
		# ta_4d = np.transpose(ta,[2,0,1])
		# ta_4d = np.reshape(ta_4d,[92,39,80,180],order = 'F')
		
		''' initial submissoin'''
		# ta_90th = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N_detrend_105_39_sig_only.mat')['hw']
		
		''' first revision '''
		ta_90th = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_ori']
		# ta_90th = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_detrend']
		ta_90th = np.transpose(ta_90th,[2,0,1])
		ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')

		hw_sum  = ta_90th
		# blo_sum = ta_4d
		blo_sum = blo_sum_weak
		# pdb.set_trace()
		if long_duration:
			hw_sum = time_filter_new(hw_sum,6)
			# pdb.set_trace()


		# ************** only consider land part ********* 
		land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
		for day in range(blo_sum.shape[0]):
			for year in range(blo_sum.shape[1]):
				blo_sum[day,year,:,:][np.where(land_mask == 0 )] = 0
				hw_sum[day,year,:,:][np.where(land_mask == 0 )] = 0

		# ************* blo related or not************* 
		# if blo_related:
		# 	hw_sum[np.where(blo_sum == 0)] = 0
		# else: 
		# 	hw_sum[np.where(blo_sum == 1)] = 0

		if monthly:
			hw_daynum_thres = np.zeros([39*3,5],dtype=np.float16)
			blo_daynum_thres = np.zeros([39*3,5],dtype=np.float16)

			hw_gridnum_thres = np.zeros([39*3,5],dtype=np.float16)
			blo_gridnum_thres = np.zeros([39*3,5],dtype=np.float16)

		else:
			hw_daynum_thres = np.zeros([39,5],dtype=np.float16)
			blo_daynum_thres = np.zeros([39,5],dtype=np.float16)
			hw_gridnum_thres = np.zeros([39,5],dtype=np.float16)
			blo_gridnum_thres = np.zeros([39,5],dtype=np.float16)

		# for grid_thres in range(0,3000,600):
		# for grid_thres in range(0,2000,400):
		for grid_thres in range(0,4000,800):
		# for grid_thres in range(0,500,100):
			if grid_thres < 3600:
				interval = 800
			else:
				interval = 10000
			hw_daynum_thres[:,np.int(grid_thres/800)] = daynum(hw_sum,np.int(grid_thres/5),np.int(interval/5), de_trend = False,thres= True, yearly = True)
			blo_daynum_thres[:,np.int(grid_thres/800)] = daynum(blo_sum,grid_thres,interval, de_trend = False,thres= True,yearly = True)

		if norm: 
			hw_daynum_thres = stats.zscore(hw_daynum_thres,axis = 1)
			blo_daynum_thres = stats.zscore(blo_daynum_thres, axis = 1)


		for day_thres in range(0,15,3):
			if day_thres < 12:
				interval = 3
			else:
				interval = 100
			
			hw_gridnum_thres[:,np.int((day_thres-0)/3)] = gridnum(hw_sum,day_thres,interval, de_trend = False,thres=True,yearly = True)
			blo_gridnum_thres[:,np.int((day_thres-0)/3)] = gridnum(blo_sum,day_thres,interval, de_trend = False,thres=True,yearly = True)

		if norm: 
			# pdb.set_trace()
			hw_gridnum_thres = stats.zscore(hw_gridnum_thres,axis = 1)
			blo_gridnum_thres = stats.zscore(blo_gridnum_thres,axis = 1)

		if ratio:
			for col in range(5):
				hw_daynum_thres[:,col] = hw_daynum_thres[:,col] / np.mean(hw_daynum_thres,axis = 0)[col]
				blo_daynum_thres[:,col] = blo_daynum_thres[:,col] / np.mean(blo_daynum_thres,axis = 0)[col]
				hw_gridnum_thres[:,col] = hw_gridnum_thres[:,col] / np.mean(hw_gridnum_thres,axis = 0)[col]
				blo_gridnum_thres[:,col] = blo_gridnum_thres[:,col] / np.mean(blo_gridnum_thres,axis = 0)[col]
		
		# pdb.set_trace()


		# ************* task 1 trend-1d*************** 

		cor_rp_hw_daynum = cor_relation_1d(hw_daynum_thres) # row-hw col-blo
		cor_rp_hw_daynum[:,0][np.where(cor_rp_hw_daynum[:,1] > 0.05)]= np.nan

		cor_rp_hw_gridnum = cor_relation_1d(hw_gridnum_thres) # row-hw col-blo
		cor_rp_hw_gridnum[:,0][np.where(cor_rp_hw_gridnum[:,1] > 0.05)]= np.nan

		cor_rp_blo_daynum = cor_relation_1d(blo_daynum_thres) # row-hw col-blo
		cor_rp_blo_daynum[:,0][np.where(cor_rp_blo_daynum[:,1] > 0.05)]= np.nan

		cor_rp_blo_gridnum = cor_relation_1d(blo_gridnum_thres) # row-hw col-blo
		cor_rp_blo_gridnum[:,0][np.where(cor_rp_blo_gridnum[:,1] > 0.05)]= np.nan

		pdb.set_trace()

		# sio.savemat(file_dir + '/temporal_result/trend/sig_trend_thres_hw_90th_3d_blo_1.0_0.35_extent_4000_15_5_detrend_sig_only.mat',{'trend_hw_day':cor_rp_hw_daynum,'trend_hw_grid':cor_rp_hw_gridnum,'trend_blo_day':cor_rp_blo_daynum,'trend_blo_grid':cor_rp_blo_gridnum})
		# sio.savemat('./hw_blo_ta/ratio_thres_hw_blo_pv_1.2_2000_15_5_valid.mat',{'ratio_hw_day':hw_daynum_thres,'ratio_hw_grid':hw_gridnum_thres,'ratio_blo_day':blo_daynum_thres,'ratio_blo_grid':blo_gridnum_thres})
		
		''' initial submission '''
		# sio.savemat(file_dir + '/temporal_result/trend/ratio_thres_hw_90th_3d_blo_pv_1.2_0.4_extent_4000_15_5_valid_nondetrend.mat',{'ratio_hw_day':hw_daynum_thres,'ratio_hw_grid':hw_gridnum_thres,'ratio_blo_day':blo_daynum_thres,'ratio_blo_grid':blo_gridnum_thres})
		
		'''fist revision'''
		# np.savez(file_dir + '/cpc_tmp/ratio_thres_hw_90th_3d_blo_pv_1.2_0.4_extent_4000_15_5_valid_nondetrend.npz',ratio_hw_day = hw_daynum_thres , ratio_hw_grid = hw_gridnum_thres, ratio_blo_day = blo_daynum_thres,ratio_blo_grid = blo_gridnum_thres)
		# np.savez(file_dir + '/cpc_tmp/ratio_thres_hw_90th_3d_blo_pv_1.2_0.4_extent_4000_15_5_valid_detrend.npz',ratio_hw_day = hw_daynum_thres , ratio_hw_grid = hw_gridnum_thres, ratio_blo_day = blo_daynum_thres,ratio_blo_grid = blo_gridnum_thres)
		# pdb.set_trace()

		# sio.savemat('./hw_blo_ta/sig_trend_thres_ta_75th_blo_1.0_ratio_4000_15_5_valid.mat',{'trend_blo_ta_day':cor_rp_hw_daynum,'trend_blo_ta_grid':cor_rp_hw_gridnum,'trend_ta_day':cor_rp_blo_daynum,'trend_ta_grid':cor_rp_blo_gridnum})
		# sio.savemat('./hw_blo_ta/ratio_thres_ta_75th_blo_weak_ratio_2000_15_5_valid.mat',{'ratio_blo_ta_day':hw_daynum_thres,'ratio_blo_ta_grid':hw_gridnum_thres,'ratio_ta_day':blo_daynum_thres,'ratio_ta_grid':blo_gridnum_thres})
		# sio.savemat('./hw_blo_ta/ori_thres_ta_75th_blo_1.0_ratio_4000_15_5_valid.mat',{'ori_blo_ta_day':hw_daynum_thres,'ori_blo_ta_grid':hw_gridnum_thres,'ori_ta_day':blo_daynum_thres,'ori_ta_grid':blo_gridnum_thres})

	if trend_of_prob:
		yearly  = True
		monthly = False
		hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
		blo_sum = blo_sum_weak
		# blo_sum = np.ones_like(blo_sum_weak) 

		# ta_90th = sio.loadmat('./ta-max/hw_tamax_90_3d_35_75N_detrend_mean.mat')['hw']
		# ta_90th = np.transpose(ta_90th,[2,0,1])
		# ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')

		# hw_sum =  ta_90th

		# # blo_ta = sio.loadmat('./hw_blo_ta/blo_pv_1.2_and_ta_75th.mat')['blo_ta']
		# # blo_sum = blo_ta
		# tamax_75th = sio.loadmat('./ta-max/tamax_dif_75th_summer_15win.mat')['ta_75th'] 
		# tamax_75th_4d = np.transpose(tamax_75th,[2,0,1])
		# tamax_75th_4d = np.reshape(tamax_75th_4d,[92,39,80,180],order = 'F')

		# tamax_80th = sio.loadmat('./ta-max/tamax_dif_80th_summer_15win.mat')['ta_80th']
		# tamax_80th_4d = np.transpose(tamax_80th,[2,0,1])
		# tamax_80th_4d = np.reshape(tamax_80th_4d,[92,39,80,180],order = 'F')

		# tamax_85th = sio.loadmat('./ta-max/tamax_dif_85th_summer_15win.mat')['ta_85th']
		# tamax_85th_4d = np.transpose(tamax_85th,[2,0,1])
		# tamax_85th_4d = np.reshape(tamax_85th_4d,[92,39,80,180],order = 'F')

		ta_90th = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N_detrend_105_39_sig_only.mat')['hw']
		tamax_90th_4d = np.transpose(ta_90th,[2,0,1])
		tamax_90th_4d = np.reshape(tamax_90th_4d,[92,39,80,180],order = 'F')

		hw_sum = tamax_90th_4d


		# blo_sum = tamax_75th_4d
		# hw_sum = tamax_90th_4d

		land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
		for day in range(blo_sum.shape[0]):
			for year in range(blo_sum.shape[1]):
				blo_sum[day,year,:,:][np.where(land_mask == 0 )] = 0
				hw_sum[day,year,:,:][np.where(land_mask == 0 )] = 0

		blo_and_hw = co_occur(hw_sum,blo_sum,False,False)
		if yearly:
			prob = np.zeros([39,1],dtype = np.float32)
			for year in range(39):
				blo_and_hw_year = np.sum(np.squeeze(blo_and_hw[:,year,:,:]))
				blo_year = np.sum(np.squeeze(blo_sum[:,year,:,:]))
				prob[year,0] = blo_and_hw_year/blo_year
		elif monthly:
			prob = np.zeros([39,3],dtype = np.float32) # 0-dim year; 1-dim month 6 7 8 
			for year in range(39):
				blo_and_hw_mon6 = np.sum(np.squeeze(blo_and_hw[0:30,year,:,:]))
				blo_mon6 = np.sum(np.squeeze(blo_sum[0:30,year,:,:]))
				prob[year,0] = blo_and_hw_mon6/blo_mon6

				blo_and_hw_mon7 = np.sum(np.squeeze(blo_and_hw[30:61,year,:,:]))
				blo_mon7 = np.sum(np.squeeze(blo_sum[30:61,year,:,:]))
				prob[year,1] = blo_and_hw_mon7/blo_mon7

				blo_and_hw_mon8 = np.sum(np.squeeze(blo_and_hw[61:,year,:,:]))
				blo_mon8 = np.sum(np.squeeze(blo_sum[61:,year,:,:]))
				prob[year,2] = blo_and_hw_mon8/blo_mon8
				print(blo_mon6,blo_mon7,blo_mon8)

			prob = np.reshape(prob,[39*3,1],order = 'F')
			# pdb.set_trace()

		if norm_prob:
			prob = (prob - np.nanmin(prob))/(np.nanmax(prob)- np.nanmin(prob))

		cor_rp_hw_blo_prob = cor_relation_1d(prob) # row-hw col-blo
		# cor_rp_hw_blo_prob[:,0][np.where(cor_rp_hw_blo_prob[:,1] > 0.05)]= np.nan
		pdb.set_trace()
		sio.savemat(file_dir + '/temporal_result/trend/blo_pv1.0_0.35_related_hw_detrend_sig_yearly_prob_trend',{'trend':cor_rp_hw_blo_prob,'prob':prob})
		# sio.savemat('./hw_blo_ta/ta_75th_ta_95th_related_ta_yearly_prob_trend',{'trend':cor_rp_hw_blo_prob,'prob':prob})
		# sio.savemat('./hw_blo_ta/ta_90th_related_ta_grid_ratio_monthly_prob_trend',{'trend':cor_rp_hw_blo_prob,'prob':prob})

		# sio.savemat('./hw_blo_ta/blo_pv1.0_related_hw_monthly_prob_trend_norm',{'trend':cor_rp_hw_blo_prob,'prob':prob})

		# ***************** correlation for probability yearly ***************** 
		# prob_hw = sio.loadmat('./hw_blo_ta/blo_related_hw_monthly_prob_trend_norm_3.mat')['prob_pv1']
		# prob_ta = sio.loadmat('./hw_blo_ta/ta_75th_to_90th_ta_grid_ratio_monthly_prob_trend_norm_4.mat')['prob_75th']
		
		# cor_rp = cor_relation_2d(prob_hw,prob_ta,order = True)
		# cor_rp[:,:,0][np.where(cor_rp[:,:,1] > 0.05)]= np.nan

		# sio.savemat('./hw_blo_ta/cor_prob_hw_in_blo_with_ta_in_grid_monthly_order.mat',{'cor_rp':cor_rp})



		# pdb.set_trace()

		# ******** tamax_yearly_mean ************* 
		# ta =  sio.loadmat('./ta-max/daily_tamax_summer_35_75N_new.mat')['ta'][:-1,:-1,:]
		# ta = np.transpose(ta,[2,0,1])
		# ta = np.reshape(ta,[92,39,80,180],order = 'F')
		# land_mask = sio.loadmat('./hw_blo_ta/land_mask_35_75N_70_160E.mat')['mask']
		# for day in range(92):
		# 	for year in range(39):
		# 		ta[day,year,:,:][np.where(land_mask == 0 )] = np.nan
		
		# ta_mean = np.zeros([39,1],dtype = np.float32)
		# for year in range(39):
		# 	ta_mean[year,0] = np.nanmean(np.squeeze(ta[:,year,:,:]))
		# 	print(ta_mean[year,0])

		# sio.savemat('./hw_blo_ta/tamax_75th_yearly_mean.mat',{'ta_mean':ta_mean})
		# pdb.set_trace()





	# ************ task 2 cor-relation 2d ***************  
	# pdb.set_trace()

	if correlation:
		if withta:
			# hw_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_blo_tm90_extent_3000_15_5_valid.mat')['ori_hw_day']
			# hw_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_blo_tm90_extent_3000_15_5_valid.mat')['ori_hw_grid']

			# hw_fea = np.append(hw_day,hw_grid,axis = 1)

			# blo_tm90_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_blo_tm90_2000_15_5_valid.mat')['ori_blo_day']
			# blo_tm90_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_blo_tm90_2000_15_5_valid.mat')['ori_blo_grid']


			# blo_weak_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_blo_pv_1.2_2000_15_5_valid.mat')['ori_blo_day']
			# blo_weak_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_blo_pv_1.2_2000_15_5_valid.mat')['ori_blo_grid']


			# # ************ 75th ************* 
			# ta_day = sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_weak_ratio_4000_15_5_valid.mat')['ori_ta_day']
			# ta_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_weak_ratio_4000_15_5_valid.mat')['ori_ta_grid']


			# ta_blo_tm90_day =  sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_tm90_ta75th_2000_15_5_valid.mat')['ori_blo_ta_day']
			# ta_blo_tm90_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_tm90_ta75th_2000_15_5_valid.mat')['ori_blo_ta_grid']

			# ta_blo_weak_day =  sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_weak_ratio_2000_15_5_valid.mat')['ori_blo_ta_day']
			# ta_blo_weak_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_weak_ratio_2000_15_5_valid.mat')['ori_blo_ta_grid']


			hw_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_5d_blo_pv_1.0_extent_2000_15_5_valid.mat')['ori_hw_day']
			hw_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_5d_blo_pv_1.0_extent_2000_15_5_valid.mat')['ori_hw_grid']

			hw_fea = np.append(hw_day,hw_grid,axis = 1)

			blo_tm90_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_blo_tm90_2000_15_5_valid.mat')['ori_blo_day']
			blo_tm90_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_blo_tm90_2000_15_5_valid.mat')['ori_blo_grid']


			blo_weak_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_5d_blo_pv_1.0_extent_2000_15_5_valid.mat')['ori_blo_day']
			blo_weak_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_5d_blo_pv_1.0_extent_2000_15_5_valid.mat')['ori_blo_grid']


			# ************ 75th ************* 
			ta_day = sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_weak_ratio_4000_15_5_valid.mat')['ori_ta_day']
			ta_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_weak_ratio_4000_15_5_valid.mat')['ori_ta_grid']


			ta_blo_tm90_day =  sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_tm90_ta75th_2000_15_5_valid.mat')['ori_blo_ta_day']
			ta_blo_tm90_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_tm90_ta75th_2000_15_5_valid.mat')['ori_blo_ta_grid']

			ta_blo_weak_day =  sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_1.0_ratio_4000_15_5_valid.mat')['ori_blo_ta_day']
			ta_blo_weak_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_75th_blo_1.0_ratio_4000_15_5_valid.mat')['ori_blo_ta_grid']

			# ************** 85th *********** 
			# ta_day = sio.loadmat('./hw_blo_ta/ori_thres_ta_85th_blo_weak_ratio_4000_15_5_valid.mat')['ori_ta_day']
			# ta_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_85th_blo_weak_ratio_4000_15_5_valid.mat')['ori_ta_grid']


			# ta_blo_tm90_day =  sio.loadmat('./hw_blo_ta/ori_thres_ta_85th_blo_tm90_ta75th_2000_15_5_valid.mat')['ori_blo_ta_day']
			# ta_blo_tm90_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_85th_blo_tm90_ta75th_2000_15_5_valid.mat')['ori_blo_ta_grid']

			# ta_blo_weak_day =  sio.loadmat('./hw_blo_ta/ori_thres_ta_85th_blo_weak_ratio_2000_15_5_valid.mat')['ori_blo_ta_day']
			# ta_blo_weak_grid = sio.loadmat('./hw_blo_ta/ori_thres_ta_85th_blo_weak_ratio_2000_15_5_valid.mat')['ori_blo_ta_grid']

			# x1 = [12, 2, 1, 12, 2]
			# x2 = [1, 4, 7, 1, 0]
			# tau, p_value = sp.stats.kendalltau(x1, x2)
			# pdb.set_trace()

			blo_ta_fea = np.concatenate((ta_day,ta_grid,blo_tm90_day,blo_tm90_grid,blo_weak_day,blo_weak_grid,ta_blo_tm90_day,ta_blo_tm90_grid,ta_blo_weak_day,ta_blo_weak_grid),axis = 1)
			sio.savemat('./hw_blo_ta/hw_blo_ta_thres_series_90th_5d_pv_1.0.mat',{'hw_fea':hw_fea,'blo_ta':blo_ta_fea})


			# cor_rp = cor_relation_2d(hw_daynum_thres,blo_daynum_thres) # row-hw col-blo
			cor_rp = cor_relation_2d(hw_fea,blo_ta_fea,order = False)
			cor_rp[:,:,0][np.where(cor_rp[:,:,1] > 0.05)]= np.nan
			sio.savemat('./hw_blo_ta/cor_hw_90th_5d_vs_blo_pv_1.0_ta_75th_r_p.mat',{'cor_rp':cor_rp})
		else:
			#### *********************** yearly ******************************* 
			# hw_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_3d_blo_tm90_extent_3000_15_5_valid_all_detrend.mat')['ori_hw_day']
			# hw_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_3d_blo_tm90_extent_3000_15_5_valid_all_detrend.mat')['ori_hw_grid']

			# hw_fea = np.append(hw_day,hw_grid,axis = 1)

			# blo_tm90_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_3d_blo_tm90_extent_3000_15_5_valid_all_detrend.mat')['ori_blo_day']
			# blo_tm90_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_3d_blo_tm90_extent_3000_15_5_valid_all_detrend.mat')['ori_blo_grid']


			# blo_weak_day = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_5d_blo_pv_1.0_extent_2000_15_5_valid.mat')['ori_blo_day']
			# blo_weak_grid = sio.loadmat('./hw_blo_ta/ori_thres_hw_90th_5d_blo_pv_1.0_extent_2000_15_5_valid.mat')['ori_blo_grid']

			# blo_fea = np.concatenate((blo_tm90_day,blo_tm90_grid,blo_weak_day,blo_weak_grid),axis = 1)
			# sio.savemat('./hw_blo_ta/hw_90th_detrend_blo_thres_series_90th_tm90_pv_1.0.mat',{'hw_fea':hw_fea,'blo_ta':blo_fea})


			#### ************************ monthly *******************************

			hw_day = sio.loadmat(file_dir + '/temporal_result/trend/ori_thres_hw_90th_3d_blo_tm90_extent_4000_15_5_valid_nondetrend.mat')['ori_hw_day']
			hw_grid = sio.loadmat(file_dir + '/temporal_result/trend/ori_thres_hw_90th_3d_blo_tm90_extent_4000_15_5_valid_nondetrend.mat')['ori_hw_grid']

			hw_fea = np.append(hw_day,hw_grid,axis = 1)

			blo_tm90_day = sio.loadmat(file_dir + '/temporal_result/trend/ori_thres_hw_90th_3d_blo_tm90_extent_4000_15_5_valid_detrend_sig_only.mat')['ori_blo_day']
			blo_tm90_grid = sio.loadmat(file_dir + '/temporal_result/trend/ori_thres_hw_90th_3d_blo_tm90_extent_4000_15_5_valid_detrend_sig_only.mat')['ori_blo_grid']


			blo_weak_day = sio.loadmat(file_dir + '/temporal_result/trend/ori_thres_hw_90th_3d_blo_pv_1.0_0.35_extent_4000_15_5_valid_detrend_sig_only.mat')['ori_blo_day']
			blo_weak_grid = sio.loadmat(file_dir + '/temporal_result/trend/ori_thres_hw_90th_3d_blo_pv_1.0_0.35_extent_4000_15_5_valid_detrend_sig_only.mat')['ori_blo_grid']

			blo_fea = np.concatenate((blo_tm90_day,blo_tm90_grid,blo_weak_day,blo_weak_grid),axis = 1)
			sio.savemat(file_dir + '/temporal_result/trend/ori_hw_blo_day_grid_num.mat',{'hw_fea':hw_fea,'blo_fea':blo_fea})



			cor_rp = cor_relation_2d(hw_fea,blo_fea,order = False)
			cor_rp[:,:,0][np.where(cor_rp[:,:,1] > 0.05)]= np.nan
			pdb.set_trace()
			sio.savemat(file_dir + '/temporal_result/trend/cor_hw_90th_nondetrend_vs_blo_tm90_pv_1.0_0.35_r_p_yearly.mat',{'cor_rp':cor_rp})

main_temporal()

