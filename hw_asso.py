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

from hw_blo_basic import pv_overlap_check,pv_overlap_check_2d,pv_thres  # for the big event detection
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from hw_blo_basic import load_data,mon_blo_freq, make_ax, daynum, gridnum, daily_gridnum, persistence, block_freq, co_occur, condi_prob, cor_relation_2d, cor_relation_1d, geomap_china,lon_mean_blocking,lag_co_occur,event_number


''' concurrent --- conditional probability '''

def main_menclo_test():
	# ********** Mento calor test *************** 
	plot_only = False
	random_repeat = 1000
	withta = False
	sign = True
	dif = False # difference
	factor = False # ratio
	day_thres = 3
	long_duration = False
	north = False
	anom_hw = True # p(hw|blocking)/(hw|summer) 
	whole_summer = True
	diff_ratio = False

	# **************** load data ***************************
	file_dir = '/home/user/Documents/research/project1'
	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	

	'''initial submission'''
	# ta_90th_3d = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N_detrend_105_39_sig_only.mat')['hw']

	''' first revision '''
	ta_90th_3d = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_detrend']
	ta_90th = np.transpose(ta_90th_3d,[2,0,1])
	ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')

	'''initial submission'''
	# pv_list = [1.3,1.2]
	# ratio_list = [0.35,0.5]

	''' first revision '''
	# pv_list = [1.2]
	# ratio_list = [0.4]
	
	# pv_list = [1.3,1.2]
	# ratio_list = [0.5,0.45,0.35]


	pv_list = [1.3]
	ratio_list = [0.4]



	# pv_list = [1.1]
	# ratio_list = [0.45]

	if not plot_only:
		for pv in pv_list:
			for ratio in ratio_list:
				blo_sum_weak_3d = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv'+str(pv)+'_5day_2dtrack_weighted_3d_35_75N_all_ratio_' + str(ratio)+'_daily_extent_100.mat')['blocking']
				blo_sum_weak = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv'+ str(pv)+'_5day_2dtrack_weighted_4d_35_75N_all_ratio_'+ str(ratio)+'_daily_extent_100.mat')['blocking']

				# blo_type = '_pv_' + str(pv) + '_'+ str(ratio) +'_daily_'
				blo_type = '_tm90_with_spatialfilter_'
				hw_type = '_tamax_90th_detrend_'
				ta_type = '_'

				hw_sum = ta_90th
				hw_sum_3d = ta_90th_3d
				
				blo_sum = blo_sum_tm90
				blo_sum_3d = blo_sum_tm90_3d
				hw_sum_3d[np.isnan(hw_sum_3d)] = 0 
				hw_sum[np.isnan(hw_sum)] = 0 

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


				''' initial submission'''
				# sio.savemat(file_dir + '/temporal_result/significant_mentocor_test_P(heatwave|blocking)' +  blo_type + hw_type + ta_type + str(random_repeat) +'.mat',{'anom':hw_freq_anom})
				# sio.savemat(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)' +  blo_type + hw_type + ta_type + str(random_repeat) +'.mat',{'anom':hw_freq_anom})

				''' first revision '''
				np.save(file_dir + '/cpc_tmp/significant_mentocor_test_P(heatwave|blocking)' +  blo_type + hw_type + ta_type + str(random_repeat) +'.npy',hw_freq_anom)

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



''' extende temporal association '''
def main_lag_ev_based():
	two_figure = True
	ev_num_dis = False
	d5 = False

	file_dir = '/home/user/Documents/research/project1'
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']

	''' initial submission '''
	# ta_90th = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N_detrend_105_39_sig_only.mat')['hw']

	''' first revision'''
	ta_90th = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_detrend']
	
	# if d5:
	# 	ta_90th = time_filter(ta_90th,5)
	# ta_90th = np.transpose(ta_90th,[2,0,1])
	# ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')

	if d5:
		ta_90th = np.transpose(ta_90th,[2,0,1])
		ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')
		ta_90th = time_filter_new(ta_90th,3) # the original time filter for d3 is for 3d, flaw
	hw_sum = ta_90th 

	blo_sum_tm90 = time_filter_new (blo_sum_tm90,5)
	blo_sum_freq = block_freq(blo_sum_tm90,dimen = 'grid')/92/39
	blo_sum_freq[np.where(land_mask==0)] = np.nan

	# clevs1 = np.arange(0,0.11,0.01)
	# outfig1 = file_dir + '/result_figure/hw_event_number.png'

	# single_data = True
	# title1  = "P(blocking|summer)"
	# geomap_china(blo_sum_freq, blo_sum_freq, clevs1,clevs1,title1,title1,outfig1,single_data)
	pv_label = sio.loadmat(file_dir +'/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.4_daily_extent_100_with_label.mat')['blocking']
	
	if ev_num_dis:
		print("true")
		# ************ event number distribution **************** 
		hw_ev_num, hw_ev_du = event_number(hw_sum, hw = True, tm = False, pv = False)
		# tm_ev_num, tm_ev_du = event_number(blo_sum_tm90, hw = False, tm = True, pv = False)
		# pv_ev_num, pv_ev_du = event_number(pv_label, hw = False, tm = False, pv = True)
		# pdb.set_trace()

		hw_ev_num[np.where(land_mask==0 )] = np.nan
		hw_ev_du[np.where(land_mask==0 )] = np.nan
		# tm_ev_num[np.where(land_mask==0 )] = np.nan
		# tm_ev_du[np.where(land_mask==0 )] = np.nan
		# pv_ev_num[np.where(land_mask==0 )] = np.nan
		# pv_ev_du[np.where(land_mask==0 )] = np.nan


		clevs1 = np.arange(0,2,0.02)
		clevs2 = np.arange(0,7.5,0.5)
		outfig1 = file_dir + '/result_figure/hw_event_number_new.png'
		outfig2 = file_dir + '/result_figure/blo_tm_event_number.png'
		outfig3 = file_dir + '/result_figure/blo_pv_event_number.png'

		outfig4 = file_dir + '/result_figure/hw_event_duration_new.png'
		outfig5 = file_dir + '/result_figure/blo_tm_event_duration.png'
		outfig6 = file_dir + '/result_figure/blo_pv_event_duration.png'

		if two_figure:
			single_data = True
			title1  = "Event number"
			title2  = "Event duration"
			geomap_china(hw_ev_num/39, hw_ev_num, clevs1,clevs1,title1,title1,outfig1,single_data)
			# geomap_china(tm_ev_num/39, tm_ev_num, clevs1,clevs1,title1,title1,outfig2,single_data)
			# geomap_china(pv_ev_num/39, pv_ev_num, clevs1,clevs1,title1,title1,outfig3,single_data)
			
			geomap_china(hw_ev_du, hw_ev_du, clevs2,clevs2,title2,title2,outfig4,single_data)
			# geomap_china(tm_ev_du, tm_ev_du, clevs2,clevs2,title2,title2,outfig5,single_data)
			# geomap_china(pv_ev_du, pv_ev_du, clevs2,clevs2,title2,title2,outfig6,single_data)
			pdb.set_trace()

	# ##********************************************************************  set the event characteristics database containing starting date, duration, and end date************************************************** 
	to_get_database = False
	if to_get_database:
		list_lag = []
		land_mask_year = np.repeat(land_mask[np.newaxis,:,:],92,axis = 0)
		# pdb.set_trace()
		for year in range(hw_sum.shape[1]):
			print(year)
			hw_year = np.squeeze(hw_sum [:,year,:,:])
			tm_year = np.squeeze(blo_sum_tm90[:,year,:,:])
			pv_year = np.squeeze(pv_label[:,year,:,:])
			hw_year[np.where(land_mask_year==0)] = np.nan
			tm_year[np.where(land_mask_year==0)] = np.nan
			pv_year[np.where(land_mask_year==0)] = np.nan
			for row in range(hw_sum.shape[2]):
				for col in range(hw_sum.shape[3]):
					# sample_hw = hw_sum[:,year,row,col]
					# sample_tm = blo_sum_tm90[:,year,row,col]
					# sample_pv = pv_label[:,year,row,col]

					sample_hw = hw_year[:,row,col]
					sample_tm = tm_year[:,row,col]
					sample_pv = pv_year[:,row,col]

					idx_hw, = np.where(sample_hw == 1)
					idx_hw.tolist()
					idx_tm, = np.where(sample_tm == 1)
					idx_tm.tolist()
					idx_pv = [np.where(sample_pv == num)[0].tolist() for num in range(1,100)]
					pv_ranges = [x for x in idx_pv if x !=[]]
					# pdb.set_trace()
					
					hw_ranges = []
					for k, g in itertools.groupby(enumerate(idx_hw), lambda x: x[1]-x[0]):
						hw_ranges.append(list(map(itemgetter(1), g)))
	    			
					tm_ranges = []
					for k, g in itertools.groupby(enumerate(idx_tm), lambda x: x[1]-x[0]):
						tm_ranges.append(list(map(itemgetter(1), g)))

					if hw_ranges != [] or tm_ranges != [] or pv_ranges != []:
						# pdb.set_trace()
						# list_grid  = [{'row':row},{'col':col}, {'year':year},{'hw_range':hw_ranges}, {'tm_range':tm_ranges},{'pv_range':pv_ranges}]
						list_grid  = [{'row':row,'col':col, 'year':year,'hw_range':hw_ranges, 'tm_range':tm_ranges,'pv_range':pv_ranges}]
						list_lag.append(list_grid)

		# pdb.set_trace()

		tm_gap = []
		pv_gap = []

		hw_tm_lag = []
		hw_pv_lag = []

		hw_num = 0	
		tm_num = 0
		pv_num = 0
		for i in range(len(list_lag)):
			tm_grid = list_lag[i][0]['tm_range']
			pv_grid = list_lag[i][0]['pv_range']
			hw_grid = list_lag[i][0]['hw_range']
			row = list_lag[i][0]['row']
			col = list_lag[i][0]['col']
			year = list_lag[i][0]['year']

			if len(tm_grid)>1:
				for j in range(1,len(tm_grid)):
					tm_gap.append([row,col,year,tm_grid[j][0]-tm_grid[j-1][-1]])
			if len(pv_grid)>1:
				for j in range(1,len(pv_grid)):
					pv_gap.append([row,col,year,pv_grid[j][0]-pv_grid[j-1][-1]])

			nearest = False
			based_on_hw = True

			if nearest == False:
				if based_on_hw:
					if len(hw_grid)>0:
						for j in range(len(hw_grid)):
							hw_num = hw_num+1
							if len(tm_grid)>0:
								for i_tm in range(len(tm_grid)):
									hw_d = hw_grid[j][-1]-hw_grid[j][0] + 1
									tm_d = tm_grid[i_tm][-1]-tm_grid[i_tm][0] + 1
									hw_s_s_tm = hw_grid[j][0] - tm_grid[i_tm][0]
									hw_e_e_tm = hw_grid[j][-1] - tm_grid[i_tm][-1]
									hw_s_e_tm = hw_grid[j][0] - tm_grid[i_tm][-1]
									# hw start - blo_end 
									hw_e_s_tm = hw_grid[j][-1] - tm_grid[i_tm][0]
									concurent_days_tm = len(set(hw_grid[j]).intersection(tm_grid[i_tm]))
									# pdb.set_trace()
									# hw end minus blocking start
									hw_tm_lag.append([hw_num, row,col,year,j,i_tm, hw_d,tm_d,hw_s_s_tm,hw_e_e_tm,hw_s_e_tm,hw_e_s_tm,concurent_days_tm])
							else:
								hw_d = hw_grid[j][-1]-hw_grid[j][0] + 1
								hw_tm_lag.append([hw_num,row,col,year,j,0, hw_d,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
								# hw_tm_lag.append([row,col,year,j,0, hw_d,100,100,100,100])
								
							if len(pv_grid)>0:
								for i_pv in range(len(pv_grid)):
									hw_d = hw_grid[j][-1]-hw_grid[j][0] + 1
									pv_d = pv_grid[i_pv][-1]-pv_grid[i_pv][0] + 1
									hw_s_s_pv = hw_grid[j][0] - pv_grid[i_pv][0]
									hw_e_e_pv = hw_grid[j][-1] - pv_grid[i_pv][-1]
									hw_s_e_pv = hw_grid[j][0] - pv_grid[i_pv][-1]
									hw_e_s_pv = hw_grid[j][-1] - pv_grid[i_pv][0]
									# pdb.set_trace()
									concurent_days_pv = len(set(hw_grid[j]).intersection(pv_grid[i_pv]))
									hw_pv_lag.append([hw_num,row,col,year,j,i_pv, hw_d,pv_d,hw_s_s_pv,hw_e_e_pv,hw_s_e_pv,hw_e_s_pv,concurent_days_pv])
							else:
								hw_d = hw_grid[j][-1]-hw_grid[j][0] + 1
								hw_pv_lag.append([hw_num,row,col,year,j,0, hw_d,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
				else:
					if len(tm_grid)>0:
						for j in range(len(tm_grid)):
							tm_num = tm_num+1
							if len(hw_grid)>0:
								for i_hw in range(len(hw_grid)):
									hw_d = hw_grid[i_hw][-1]-hw_grid[i_hw][0] + 1
									tm_d = tm_grid[j][-1]-tm_grid[j][0] + 1
									hw_s_s_tm = hw_grid[i_hw][0] - tm_grid[j][0]
									hw_e_e_tm = hw_grid[i_hw][-1] - tm_grid[j][-1]
									hw_s_e_tm = hw_grid[i_hw][0] - tm_grid[j][-1]
									# hw start - blo_end 
									hw_e_s_tm = hw_grid[i_hw][-1] - tm_grid[j][0]
									# pdb.set_trace()
									concurent_days_tm = len(set(hw_grid[i_hw]).intersection(tm_grid[j]))
									# hw end minus blocking start
									hw_tm_lag.append([tm_num, row,col,year,j,i_hw, hw_d,tm_d,hw_s_s_tm,hw_e_e_tm,hw_s_e_tm,hw_e_s_tm,concurent_days_tm])
							else:
								tm_d = tm_grid[j][-1]-tm_grid[j][0] + 1
								hw_tm_lag.append([tm_num,row,col,year,j,0, np.nan,tm_d,np.nan,np.nan,np.nan,np.nan,np.nan])
								# hw_tm_lag.append([row,col,year,j,0, hw_d,100,100,100,100])
								
					if len(pv_grid)>0:
						for j in range(len(pv_grid)):
							pv_num = pv_num+1
							if len(hw_grid)>0:
								for i_hw in range(len(hw_grid)):
									hw_d = hw_grid[i_hw][-1]-hw_grid[i_hw][0] + 1
									pv_d = pv_grid[j][-1]-pv_grid[j][0] + 1
									hw_s_s_pv = hw_grid[i_hw][0] - pv_grid[j][0]
									hw_e_e_pv = hw_grid[i_hw][-1] - pv_grid[j][-1]
									hw_s_e_pv = hw_grid[i_hw][0] - pv_grid[j][-1]
									# hw start - blo_end 
									hw_e_s_pv = hw_grid[i_hw][-1] - pv_grid[j][0]
									# pdb.set_trace()
									concurent_days_pv = len(set(hw_grid[i_hw]).intersection(set(pv_grid[j])))
									# pdb.set_trace()
									# hw end minus blocking start
									hw_pv_lag.append([pv_num, row,col,year,j,i_hw, hw_d,pv_d,hw_s_s_pv,hw_e_e_pv,hw_s_e_pv,hw_e_s_pv,concurent_days_pv])
							else:
								pv_d = pv_grid[j][-1]-pv_grid[j][0] + 1
								hw_pv_lag.append([pv_num,row,col,year,j,0, np.nan,pv_d,np.nan,np.nan,np.nan,np.nan,np.nan])
			
		# # pv_gap_only = np.asarray(pv_gap)[:,3] 
		# # tm_gap_only = np.asarray(tm_gap)[:,3] 
		# # idx_pv = np.where(pv_gap_only == 5)
		# # print(idx_pv[0].shape[0])
		# # idx_tm = np.where(tm_gap_only == 5)
		# # print(idx_tm[0].shape[0])
		# # pdb.set_trace()

		# # plt.hist(np.asarray(tm_gap),20,density = True,facecolor='gray', alpha=0.75)
		# # plt.xlabel('Gap between adjacent blocking events - TM')
		# # plt.ylabel('Probability')
		# # plt.xlim(0, 90)
		# # plt.ylim(0, 0.05)
		# # plt.show()

		# # plt.hist(np.asarray(pv_gap),20,density = True,facecolor='gray', alpha=0.75)
		# # plt.xlabel('Gap between adjacent blocking events - PV')
		# # plt.ylabel('Probability')
		# # plt.xlim(0, 90)
		# # plt.ylim(0, 0.05)
		# # plt.show()

		hw_tm_lag_arr = np.asarray(hw_tm_lag)
		hw_pv_lag_arr = np.asarray(hw_pv_lag)

		# # '''initial submission'''
		# # sio.savemat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_tm_lag_arr_nonnearest_all_based_on_blo_new.mat',{'tm_lag':hw_tm_lag_arr})
		# # sio.savemat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_pv_lag_arr_nonnearest_all_based_on_blo_new.mat',{'pv_lag':hw_pv_lag_arr})


		# '''first revision'''
		# np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_blo_new.npy',hw_tm_lag_arr) 
		# np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_blo_new.npy',hw_pv_lag_arr)


		# np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_blo_hwd5_new.npy',hw_tm_lag_arr) # the result based on the hw or the blo should be the same
		# np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_blo_hwd5_new.npy',hw_pv_lag_arr)


		# np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_hwd3_new.npy',hw_tm_lag_arr) 
		# np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_hwd3_new.npy',hw_pv_lag_arr)


		np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_hwd3_valid.npy',hw_tm_lag_arr) # the original time filter is based on 3D-- wrong
		np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_hwd3_valid.npy',hw_pv_lag_arr)
		pdb.set_trace()


	# pdb.set_trace()

	# ****************************** #
	'''first revision'''
	# hw_tm_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_blo.npy')
	# hw_pv_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_blo.npy')

	# hw_tm_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_blo_hwd5_new.npy')
	# hw_pv_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_blo_hwd5_new.npy')

	
	''' date 1202 '''
	# hw_tm_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_hwd3_new.npy')
	# hw_pv_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_hwd3_new.npy')


	hw_tm_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_hwd3_valid.npy')
	hw_pv_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_hwd3_valid.npy')

	# hw_tm_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_blo_new.npy')
	# hw_pv_lag_arr = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_blo_new.npy')

	# hw_tm_lag_arr  = sio.loadmat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_tm_lag_arr_nonnearest_all_based_on_blo_new.mat')['tm_lag']
	# hw_pv_lag_arr  = sio.loadmat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_pv_lag_arr_nonnearest_all_based_on_blo_new.mat')['pv_lag']

	'''association based on the dataset'''
	# 1) with timeoverlap but beyound the selected gap range\
	conperct = False
	if conperct:
		''' the percentage for concourrent association '''
		blf_num = np.load(file_dir + '/cpc_tmp/number_of_bls_subreg_new.npz')['hw_du_num']
		hwf_num = np.load(file_dir + '/cpc_tmp/number_of_hws_subreg_new.npz')['hw_du_num']
		hw_and_tm = co_occur(hw_sum,blo_sum_tm90,False,False)
		hw_and_pv = co_occur(hw_sum,blo_sum_weak,False,False)

		hw_blo_con = np.zeros([3,4],dtype = np.float16)

		sub_region = [[0,40],[40,60],[60,80]]
		for subreg in range(3):
			sample_tm = hw_and_tm[:,:,sub_region[subreg][0]:sub_region[subreg][1],:]
			sample_pv = hw_and_pv[:,:,sub_region[subreg][0]:sub_region[subreg][1],:]
			hw_blo_con[subreg,0] = np.nansum(sample_tm[~np.isnan(sample_tm)])/hwf_num[subreg,0]
			hw_blo_con[subreg,1] = np.nansum(sample_tm[~np.isnan(sample_tm)])/blf_num[subreg,0]
			hw_blo_con[subreg,2] = np.nansum(sample_pv[~np.isnan(sample_pv)])/hwf_num[subreg,1]
			hw_blo_con[subreg,3] = np.nansum(sample_pv[~np.isnan(sample_pv)])/blf_num[subreg,1]
		pdb.set_trace()


	ovlp_perct = False
	lag0 = -4
	lag1 = 4
	if ovlp_perct:
		con_tm = hw_tm_lag_arr[:,-1]
		con_pv = hw_pv_lag_arr[:,-1]

		hw_tm_ss = hw_tm_lag_arr[:,8]
		hw_tm_ee = hw_tm_lag_arr[:,9]

		hw_pv_ss = hw_pv_lag_arr[:,8]
		hw_pv_ee = hw_pv_lag_arr[:,9]

		id_hw_tm = hw_tm_lag_arr[:,0]
		id_hw_pv = hw_pv_lag_arr[:,0]


		idx_nan_tm = np.where((~np.isnan(con_tm)) & ( ~ np.isnan(hw_tm_ss)) & (~np.isnan(hw_tm_ee))) 
		idx_nan_pv = np.where((~np.isnan(con_pv)) & ( ~ np.isnan(hw_pv_ss)) & (~np.isnan(hw_pv_ee))) 
		# pdb.set_trace()
		
		hw_tm_lag_arr = hw_tm_lag_arr[idx_nan_tm[0],:]
		hw_pv_lag_arr = hw_pv_lag_arr[idx_nan_pv[0],:]


		con_tm_bl = hw_tm_lag_arr_bl[:,-1]
		con_pv_bl = hw_pv_lag_arr_bl[:,-1]

		hw_tm_ss_bl = hw_tm_lag_arr_bl[:,8]
		hw_tm_ee_bl = hw_tm_lag_arr_bl[:,9]

		hw_pv_ss_bl = hw_pv_lag_arr_bl[:,8]
		hw_pv_ee_bl = hw_pv_lag_arr_bl[:,9]

		id_hw_tm_bl = hw_tm_lag_arr_bl[:,0]
		id_hw_pv_bl = hw_pv_lag_arr_bl[:,0]


		idx_nan_tm_bl = np.where((~np.isnan(con_tm_bl)) & ( ~ np.isnan(hw_tm_ss_bl)) & (~np.isnan(hw_tm_ee_bl))) 
		idx_nan_pv_bl = np.where((~np.isnan(con_pv_bl)) & ( ~ np.isnan(hw_pv_ss_bl)) & (~np.isnan(hw_pv_ee_bl))) 
		# pdb.set_trace()
		
		hw_tm_lag_arr_bl = hw_tm_lag_arr_bl[idx_nan_tm_bl[0],:]
		hw_pv_lag_arr_bl = hw_pv_lag_arr_bl[idx_nan_pv_bl[0],:]


		hw_tm = pd.DataFrame(hw_tm_lag_arr, columns = ['hw_num', 'row','col','year','j','i_hw', 'hw_d','tm_d','hw_s_s_tm','hw_e_e_tm','hw_s_e_tm','hw_e_s_tm','concurent_days_tm'])
		tm_hw = pd.DataFrame(hw_tm_lag_arr_bl, columns = ['tm_num', 'row','col','year','j','i_hw', 'hw_d','tm_d','hw_s_s_tm','hw_e_e_tm','hw_s_e_tm','hw_e_s_tm','concurent_days_tm'])
		# pdb.set_trace()


		con_tm = hw_tm_lag_arr[:,-1]
		con_pv = hw_pv_lag_arr[:,-1]

		hw_tm_ss = hw_tm_lag_arr[:,8]
		hw_tm_ee = hw_tm_lag_arr[:,9]

		hw_pv_ss = hw_pv_lag_arr[:,8]
		hw_pv_ee = hw_pv_lag_arr[:,9]

		id_hw_tm = hw_tm_lag_arr[:,0]
		id_hw_pv = hw_pv_lag_arr[:,0]



		con_tm_bl = hw_tm_lag_arr_bl[:,-1]
		con_pv_bl = hw_pv_lag_arr_bl[:,-1]

		hw_tm_ss_bl = hw_tm_lag_arr_bl[:,8]
		hw_tm_ee_bl = hw_tm_lag_arr_bl[:,9]

		hw_pv_ss_bl = hw_pv_lag_arr_bl[:,8]
		hw_pv_ee_bl = hw_pv_lag_arr_bl[:,9]

		id_hw_tm_bl = hw_tm_lag_arr_bl[:,0]
		id_hw_pv_bl = hw_pv_lag_arr_bl[:,0]



		overlap_but_unselected_tm = np.where((con_tm > 0) & ((hw_tm_ss < lag0) | (hw_tm_ss > lag1)| (hw_tm_ee < lag0) | (hw_tm_ee > lag1)))
		overlap_but_unselected_pv = np.where((con_pv > 0) & ((hw_pv_ss < lag0) | (hw_pv_ss > lag1)| (hw_pv_ee < lag0) | (hw_pv_ee > lag1)))

		overlap_but_unselected_tm_bl = np.where((con_tm_bl > 0) & ((hw_tm_ss_bl < lag0) | (hw_tm_ss_bl > lag1)| (hw_tm_ee_bl < lag0) | (hw_tm_ee_bl > lag1)))
		overlap_but_unselected_pv_bl = np.where((con_pv_bl > 0) & ((hw_pv_ss_bl < lag0) | (hw_pv_ss_bl > lag1)| (hw_pv_ee_bl < lag0) | (hw_pv_ee_bl > lag1)))

		unique_hw_tm = np.unique(id_hw_tm[overlap_but_unselected_tm[0]])
		unique_hw_pv = np.unique(id_hw_pv[overlap_but_unselected_pv[0]])

		unique_tm_bl = np.unique(id_hw_tm_bl[overlap_but_unselected_tm_bl[0]])
		unique_pv_bl = np.unique(id_hw_pv_bl[overlap_but_unselected_pv_bl[0]])


		bls_num = np.load(file_dir + '/cpc_tmp/number_of_bls_subreg_new.npz')['hw_num']
		hws_num = np.load(file_dir + '/cpc_tmp/number_of_hws_subreg_new.npz')['hw_num']

		blf_num = np.load(file_dir + '/cpc_tmp/number_of_bls_subreg_new.npz')['hw_du_num']
		hwf_num = np.load(file_dir + '/cpc_tmp/number_of_hws_subreg_new.npz')['hw_du_num']



		# the overlap but not included percentage for manuscript 
		tms = unique_tm_bl.shape[0]/np.sum(bls_num,axis = 0)[0]
		hws_tm = unique_hw_tm.shape[0]/np.sum(hws_num,axis = 0)[0]

		pvs = unique_pv_bl.shape[0]/np.sum(bls_num,axis = 0)[1]
		hws_pv = unique_hw_pv.shape[0]/np.sum(hws_num,axis = 0)[0]
		pdb.set_trace()




	''' first revision '''
	''' check the different duration for heatwave and blocking '''

	hw_bl_duration =  False
	if hw_bl_duration:
		sub_region = [[0,40],[40,60],[60,80]]

		hw_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_hwd3_valid.npy')
		hw_tm_lag_arr  = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_nonnearest_all_based_on_blo_new.npy')
		hw_pv_lag_arr = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_nonnearest_all_based_on_blo_new.npy')

		tm_du = [1]
		pv_du = [1]
		hw_du = [1]

		hw_row_tm = hw_tm_lag_arr[:,1]
		hw_row_pv = hw_pv_lag_arr[:,1]
		# hw_row = hw_pv_lag_arr[:,1]
		hw_row = hw_lag_arr[:,1]

		hw_du_tm = hw_tm_lag_arr[:,7] # blocking duration
		hw_du_pv = hw_pv_lag_arr[:,7]
		hw_du_hw = hw_lag_arr[:,6]
		# hw_du_hw = hw_pv_lag_arr[:,6]

		# idx_nan_tm = ~np.isnan(hw_du_tm)
		# idx_nan_pv = ~np.isnan(hw_du_pv)
		# idx_nan_hw = ~np.isnan(hw_du_hw)
		
		# hw_du_tm = hw_du_tm[idx_nan_tm]
		# hw_du_pv = hw_du_pv[idx_nan_pv]
		# hw_du_hw = hw_du_hw[idx_nan_hw]
		# pdb.set_trace()

		# hw_row_tm = hw_row_tm[idx_nan_tm]
		# hw_row_pv = hw_row_pv[idx_nan_pv]
		# hw_row_hw = hw_row[idx_nan_hw]
		# pdb.set_trace()
		related = False
		for reg in range(3):
			if not related:
				tm_du_reg = hw_du_tm[np.where((hw_row_tm >= sub_region[reg][0]) & (hw_row_tm < sub_region[reg][1]))]
				pv_du_reg = hw_du_pv[np.where((hw_row_pv >= sub_region[reg][0]) & (hw_row_pv < sub_region[reg][1]))]
				hw_du_reg = hw_du_hw[np.where((hw_row >= sub_region[reg][0]) & (hw_row < sub_region[reg][1]))]
			else:

				hw_tm_ss = hw_tm_lag_arr[:,8]
				hw_tm_ee = hw_tm_lag_arr[:,9]

				hw_pv_ss = hw_pv_lag_arr[:,8]
				hw_pv_ee = hw_pv_lag_arr[:,9]


				tm_du_reg = hw_du_tm[np.where((hw_row_tm >= sub_region[reg][0]) & (hw_row_tm < sub_region[reg][1]) & (hw_tm_ss >= -1) & (hw_tm_ss < 1) & (hw_tm_ee >= -1) & (hw_tm_ss < 1))]
				pv_du_reg = hw_du_pv[np.where((hw_row_pv >= sub_region[reg][0]) & (hw_row_pv < sub_region[reg][1]) & (hw_pv_ss >= -1) & (hw_pv_ss < 1) & (hw_pv_ee >= -1) & (hw_pv_ss < 1))]
				hw_du_reg = hw_du_hw[np.where((hw_row >= sub_region[reg][0]) & (hw_row < sub_region[reg][1]) & (hw_pv_ss >= -1) & (hw_pv_ss < 1) & (hw_pv_ee >= -1) & (hw_pv_ss < 1))]
				pdb.set_trace()

			tm_du.append(tm_du_reg)
			pv_du.append(pv_du_reg)
			hw_du.append(hw_du_reg)

		tm_du = tm_du[1:]
		pv_du = pv_du[1:]
		hw_du = hw_du[1:]

		fig = plt.figure(figsize=(15, 5))
		plt.rcParams["font.family"] = "serif"
		gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])


		bxplt = True
		if bxplt: 
			ax0 = plt.subplot(gs[1])
			ax0.boxplot(tm_du)
			ax0.set_title('Duration of TMS')
			ax0.set_xticklabels(['North','Middle','South'])
			ax0.tick_params(labelsize = 12)
			plt.ylabel('Duration (d)',fontsize = 12)
			plt.ylim([0,30])
			ax0.set_yticks(np.arange(0,33,3))


			ax1 = plt.subplot(gs[2])
			ax1.boxplot(pv_du)
			ax1.set_title('Duration of PVS')
			ax1.set_xticklabels(['North','Middle','South'])
			ax1.tick_params(labelsize = 12)
			plt.ylabel('Duration (d)', fontsize = 12)
			plt.ylim([0,30])
			ax1.set_yticks(np.arange(0,33,3))

			ax2 = plt.subplot(gs[0])
			ax2.boxplot(hw_du)
			ax2.set_title('Duration of HWS')
			ax2.set_xticklabels(['North','Middle','South'])
			ax2.tick_params(labelsize = 12)
			plt.ylabel('Duration (d)', fontsize = 12)
			plt.ylim([0,30])
			ax2.set_yticks(np.arange(0,33,3))

			plt.tight_layout()
			plt.show()

			fig.savefig(file_dir + '/paper_figure/cpc_hws_bls_duration_valid_rangeall_7.pdf')
			pdb.set_trace()
		else:
			ax0 = plt.subplot(gs[1])
			plt.hist(tm_du,normed = True, bins = range(20), color = ['grey','#E69F00','#56B4E9'], label = ['North','Middle','South'])
			ax0.set_title('Duration of TMS')
			ax0.tick_params(labelsize = 12)
			plt.xlabel('Duration (d)',fontsize = 12)
			plt.ylabel('Normalize number (d)',fontsize = 12)
			plt.xlim([0,20])
			plt.legend()

			ax1 = plt.subplot(gs[2])
			plt.hist(pv_du,normed = True, bins = range(20), color = ['grey','#E69F00','#56B4E9'], label = ['North','Middle','South'])
			ax1.set_title('Duration of PVS')
			ax1.tick_params(labelsize = 12)
			plt.xlabel('Duration (d)',fontsize = 12)
			plt.ylabel('Normalize number (d)',fontsize = 12)
			plt.xlim([0,20])
			plt.legend()

			ax2 = plt.subplot(gs[0])
			plt.hist(hw_du,normed = True, bins = range(20), color = ['grey','#E69F00','#56B4E9'], label = ['North','Middle','South'])
			ax2.set_title('Duration of HWS')
			ax2.tick_params(labelsize = 12)
			plt.xlabel('Duration (d)',fontsize = 12)
			plt.ylabel('Normalize number (d)',fontsize = 12)
			plt.xlim([0,20])
			plt.legend()

			plt.tight_layout()
			plt.show()
			# pdb.set_trace()


			# sns.distplot(tm_du[0],hist = True,kde = False, label = 'North')
			# sns.distplot(tm_du[1],hist = True,kde = False, label = 'Middle')
			# sns.distplot(tm_du[2],hist = True,kde = False, label = 'South')

			fig.savefig(file_dir + '/paper_figure/cpc_hws_bls_duration_valid_range1_hist_hw_related_to_pv.pdf')

			# fig.savefig(file_dir + '/paper_figure/cpc_hws_bls_duration_valid_rangeall_hist.pdf')
			pdb.set_trace()


	qua4 = False
	qua4_beaf = np.zeros([6,4],dtype = np.float32)
	if qua4:
		for reg in range(3):
			tm = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_' +str(reg) + '_hwd5_ct.npy')
			pv = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_' +str(reg) + '_hwd5_ct.npy')
			
			# qua4_beaf[reg,0] = np.sum(np.squeeze(np.sum(tm[:8,7:])))
			# qua4_beaf[reg,1] = np.sum(np.squeeze(np.sum(tm[:8,:7])))
			# qua4_beaf[reg,2] = np.sum(np.squeeze(np.sum(tm[8:,:7])))
			# qua4_beaf[reg,3] = np.sum(np.squeeze(np.sum(tm[8:,7:])))

			# qua4_beaf[reg+3,0] = np.sum(np.squeeze(np.sum(pv[:8,7:])))
			# qua4_beaf[reg+3,1] = np.sum(np.squeeze(np.sum(pv[:8,:7])))
			# qua4_beaf[reg+3,2] = np.sum(np.squeeze(np.sum(pv[8:,:7])))
			# qua4_beaf[reg+3,3] = np.sum(np.squeeze(np.sum(pv[8:,7:])))

			'''corrected Gap_ss and Gap_ee with new quadrant'''
			qua4_beaf[reg,0] = np.sum(np.squeeze(np.sum(tm[:7,8:])))
			qua4_beaf[reg,1] = np.sum(np.squeeze(np.sum(tm[:7,:8])))
			qua4_beaf[reg,2] = np.sum(np.squeeze(np.sum(tm[7:,:8])))
			qua4_beaf[reg,3] = np.sum(np.squeeze(np.sum(tm[7:,8:])))

			qua4_beaf[reg+3,0] = np.sum(np.squeeze(np.sum(pv[:7,8:])))
			qua4_beaf[reg+3,1] = np.sum(np.squeeze(np.sum(pv[:7,:8])))
			qua4_beaf[reg+3,2] = np.sum(np.squeeze(np.sum(pv[7:,:8])))
			qua4_beaf[reg+3,3] = np.sum(np.squeeze(np.sum(pv[7:,8:])))
		pdb.set_trace()







	# df_ovlp_tm = pd.DataFrame(hw_tm_lag_arr[overlap_but_unselected_tm[0]], columns = ['tm_num', 'row','col','year','j','i_hw', 'hw_d','tm_d','hw_s_s_tm','hw_e_e_tm','hw_s_e_tm','hw_e_s_tm','concurent_days_tm'])

	
	# pdb.set_trace()

	# hw_tm_lag_arr  = sio.loadmat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_tm_lag_arr_nonnearest_all_based_on_blo.mat')['tm_lag']
	# hw_pv_lag_arr  = sio.loadmat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_pv_lag_arr_nonnearest_all_based_on_blo.mat')['pv_lag']

	# hw_tm_lag_arr  = sio.loadmat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_tm_lag_arr_nonnearest_all.mat')['tm_lag']
	# hw_pv_lag_arr  = sio.loadmat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_pv_lag_arr_nonnearest_all.mat')['pv_lag']

	# pdb.set_trace()
	# idx_tm = np.where((hw_tm_lag_arr[:,9] > 15) | (hw_tm_lag_arr[:,9] < -30))
	# idx_pv = np.where((hw_pv_lag_arr[:,9] > 15) | (hw_pv_lag_arr[:,9] < -30))
	# hw_tm_lag_arr = np.delete(hw_tm_lag_arr,idx_tm, axis = 0)
	# hw_pv_lag_arr = np.delete(hw_pv_lag_arr,idx_pv, axis = 0)
	# pdb.set_trace()

	# pdb.set_trace()
	
	# heatmap
	hw_tm_ss = hw_tm_lag_arr[:,8]
	hw_tm_ee = hw_tm_lag_arr[:,9]
	hw_tm_se = hw_tm_lag_arr[:,10]
	hw_tm_es = hw_tm_lag_arr[:,11]

	hw_pv_ss = hw_pv_lag_arr[:,8]
	hw_pv_ee = hw_pv_lag_arr[:,9]
	hw_pv_se = hw_pv_lag_arr[:,10]
	hw_pv_es = hw_pv_lag_arr[:,11]

	hw_row_tm = hw_tm_lag_arr[:,1]
	hw_col_tm = hw_tm_lag_arr[:,2]

	hw_row_pv = hw_pv_lag_arr[:,1]
	hw_col_pv = hw_pv_lag_arr[:,2]

	hw_year_tm = hw_tm_lag_arr[:,3]
	hw_year_pv = hw_pv_lag_arr[:,3]


	sub_region = [[0,40],[40,60],[60,80]]
	region_name =[ 'north','middle','south']

	# hw_tm_lag_2d = np.zeros([30,30],dtype = np.float16)
	# hw_pv_lag_2d = np.zeros([30,30],dtype = np.float16)

	# # unique heatwave number in each grids
	# hw_tm_uni_2d = np.zeros([30,30],dtype = np.float16)
	# hw_pv_uni_2d = np.zeros([30,30],dtype = np.float16)

	hw_id_tm = hw_tm_lag_arr[:,0]
	hw_id_pv = hw_pv_lag_arr[:,0]

	hw_du_tm = hw_tm_lag_arr[:,6] # heatwave duration 
	hw_du_pv = hw_pv_lag_arr[:,6]

	# hw_du_tm = hw_tm_lag_arr[:,7] # blocking duration
	# hw_du_pv = hw_pv_lag_arr[:,7]


	# heatwave number for each region
	regnum = True

	if regnum:
		hw_num_sub_region = np.zeros([3,2],dtype = np.float32)
		hw_num_du_sub_region = np.zeros([3,2],dtype = np.float32)
		for i in range(3):
			idx_tm_reg, = np.where((hw_row_tm >= sub_region[i][0]) & (hw_row_tm < sub_region[i][-1]))
			idx_tm_reg = [i for i in idx_tm_reg]
			hw_tm =  [idx for idx in hw_id_tm[idx_tm_reg]] # HW ID
			hw_num_sub_region[i,0] = len(np.unique(hw_tm))

			hw_du_tm_reg = [idx for idx in hw_du_tm[idx_tm_reg]] # HW Duration
			hw_id_du_tm = np.append(np.asarray(hw_tm)[:,np.newaxis],np.asarray(hw_du_tm_reg)[:,np.newaxis],axis = 1)
			df_hw_tm = pd.DataFrame(hw_id_du_tm,columns=['hw_id','hw_du'])
			df2_hw_tm = df_hw_tm.groupby('hw_id').apply(lambda x: x['hw_du'].unique())
			# pdb.set_trace()
			hw_num_du_sub_region[i,0] = df2_hw_tm.sum()

			idx_pv_reg, = np.where((hw_row_pv >= sub_region[i][0]) & (hw_row_pv < sub_region[i][-1]))
			idx_pv_reg = [i for i in idx_pv_reg]
			hw_pv =  [idx for idx in hw_id_pv[idx_pv_reg]]
			hw_num_sub_region[i,1] = len(np.unique(hw_pv))

			
			hw_du_pv_reg = [idx for idx in hw_du_pv[idx_pv_reg]] # HW Duration
			hw_id_du_pv = np.append(np.asarray(hw_pv)[:,np.newaxis],np.asarray(hw_du_pv_reg)[:,np.newaxis],axis = 1)
			df_hw_pv = pd.DataFrame(hw_id_du_pv,columns=['hw_id','hw_du'])
			df2_hw_pv = df_hw_pv.groupby('hw_id').apply(lambda x: x['hw_du'].unique())
			hw_num_du_sub_region[i,1] = df2_hw_pv.sum()

			# hw_du = [idx for idx in hw_duration_tm[idx_tm]]
			# hw_id_times_du_tm = [a*b for a,b in zip(hw_tm,hw_du)]
			# hw_id_du_tm = np.append(np.asarray(hw_tm)[:,np.newaxis],np.asarray(hw_id_times_du_tm)[:,np.newaxis],axis = 1)
			# hw_num_du_sub_region[i,0] = np.sum(hw_id_du_tm,axis=0)[:,1]
			# hw_id_times_du_tm = [a*b for a,b in zip(hw_tm,hw_du)]
			# hw_id_du_tm = np.append(np.asarray(hw_tm)[:,np.newaxis],np.asarray(hw_id_times_du_tm)[:,np.newaxis],axis = 1)

	hw_sum_whole = np.sum(hw_num_sub_region,axis =0)
	hw_du_sum_whole = np.sum(hw_num_du_sub_region,axis =0)
	# pdb.set_trace()

	# np.savez(file_dir + '/cpc_tmp/number_of_bls_subreg_new.npz',hw_num = hw_num_sub_region,hw_du_num = hw_num_du_sub_region)
	# np.savez(file_dir + '/cpc_tmp/number_of_hws_subreg_new.npz',hw_num = hw_num_sub_region,hw_du_num = hw_num_du_sub_region)

	hw_id_tm_l = []
	hw_id_pv_l = []

	whole_region = False
	du_wei = False
	ct = True
	
	# range0 = -15
	# range1 = 15

	range0 = -7
	range1 = 7

	grid_interval = 1
	row_num = (range1-range0)/grid_interval + 1 # from -7 to 7 15number 
	row_num = np.int(row_num)

 
	if whole_region:
		hw_tm_lag_2d = np.zeros([row_num,row_num],dtype = np.float16)
		hw_pv_lag_2d = np.zeros([row_num,row_num],dtype = np.float16)

		# unique heatwave number counting in each grids
		hw_tm_uni_2d_ct = np.zeros([row_num,row_num],dtype = np.float32)
		hw_pv_uni_2d_ct = np.zeros([row_num,row_num],dtype = np.float32)

		
		# unique heatwave number percentage in each grids
		hw_tm_uni_2d = np.zeros([row_num,row_num],dtype = np.float16)
		hw_pv_uni_2d = np.zeros([row_num,row_num],dtype = np.float16)

		# unique heatwave weighted by duration
		hw_tm_uni_du_2d = np.zeros([row_num,row_num],dtype = np.float16)
		hw_pv_uni_du_2d = np.zeros([row_num,row_num],dtype = np.float16)


		for row in range(row_num):
			for col in range(row_num):
				# print(col)
			# -30 ~ 30
			# x-cordinate then use row, if y use col
				idx_ss_tm, = np.where((hw_tm_ss < range0 + (row+1)*grid_interval) & (hw_tm_ss >= range0 + (row)*grid_interval))
				idx_ee_tm, = np.where((hw_tm_ee < range0 + (col+1)*grid_interval) & (hw_tm_ee >= range0 + (col)*grid_interval))

				idx_es_tm, = np.where((hw_tm_es < range0 + (col+1)*grid_interval) & (hw_tm_es >= range0 + (col)*grid_interval))
				idx_se_tm, = np.where((hw_tm_se < range0 + (row+1)*grid_interval) & (hw_tm_se >= range0 + (row)*grid_interval))


				idx_ss_pv, = np.where((hw_pv_ss < range0 + (row+1)*grid_interval) & (hw_pv_ss >= range0+ (row)*grid_interval))
				idx_ee_pv, = np.where((hw_pv_ee < range0 + (col+1)*grid_interval) & (hw_pv_ee >= range0 + (col)*grid_interval))

				idx_es_pv, = np.where((hw_pv_es < range0+ (col+1)*grid_interval) & (hw_pv_es >= range0 + (col)*grid_interval))
				idx_se_pv, = np.where((hw_pv_se < range0 + (row+1)*grid_interval) & (hw_pv_se >= range0 + (row)*grid_interval))
				
				idx_tm_set = set(idx_ss_tm).intersection(set(idx_ee_tm))
				# idx_tm = set(idx_es_tm).intersection(set(idx_se_tm)) 
				hw_tm_lag_2d[row,col] = len(idx_tm_set) # ununique counting
				
				idx_tm = [i for i in idx_tm_set]
				hw_1 =  [idx for idx in hw_id_tm[idx_tm]]
				hw_tm_uni_2d_ct [row,col]= len(np.unique(hw_1)) 
				hw_tm_uni_2d[row,col] = len(np.unique(hw_1))/hw_sum_whole[0] # unique percentage
				hw_id_tm_l.append(hw_1)
				
				hw_du_1 = [idx for idx in hw_du_tm[idx_tm]]
				hw_id_du_1 = np.append(np.asarray(hw_1)[:,np.newaxis],np.asarray(hw_du_1)[:,np.newaxis],axis = 1)
				df_hw_1 = pd.DataFrame(hw_id_du_1,columns=['hw_id','hw_du'])
				df2_hw_1 = df_hw_1.groupby('hw_id').apply(lambda x: x['hw_du'].unique())
				# print('dataframe:',df2_hw_1,'sum:',df2_hw_1.sum())
				if df2_hw_1.empty == False:
					# hw_tm_uni_du_2d[row,col] = df2_hw_1.sum()/hw_du_sum_whole[0]
					hw_tm_uni_du_2d[row,col] = df2_hw_1.sum()/3000
				# pdb.set_trace()

				# idx_pv = set(idx_es_pv).intersection(set(idx_se_pv)) 
				idx_pv_set = set(idx_ss_pv).intersection(set(idx_ee_pv)) 
				hw_pv_lag_2d[row,col] = len(idx_pv_set)

				idx_pv = [i for i in idx_pv_set]
				hw_2 =  [idx for idx in hw_id_pv[idx_pv]]
				hw_pv_uni_2d_ct[row,col] = len(np.unique(hw_2)) 
				hw_pv_uni_2d[row,col] = len(np.unique(hw_2))/hw_sum_whole[1]
				hw_id_pv_l.append(hw_2)

				hw_du_2 = [idx for idx in hw_du_pv[idx_pv]]
				hw_id_du_2 = np.append(np.asarray(hw_2)[:,np.newaxis],np.asarray(hw_du_2)[:,np.newaxis],axis = 1)
				df_hw_2 = pd.DataFrame(hw_id_du_2,columns=['hw_id','hw_du'])
				df2_hw_2 = df_hw_2.groupby('hw_id').apply(lambda x: x['hw_du'].unique())
				if df2_hw_2.empty == False:
					# hw_pv_uni_du_2d[row,col] = df2_hw_2.sum()/hw_du_sum_whole[1]
					hw_pv_uni_du_2d[row,col] = df2_hw_2.sum()/3000

		# sio.savemat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_pv_lag_range_15_15_unique_blo_ct.mat',{'tm':hw_tm_uni_2d_ct,'pv':hw_pv_uni_2d_ct})
		'''first revision'''
		np.savez(file_dir + '/cpc_tmp/hw_blo_lag_range_based_on_blo_7_whole_region_ct_new.npz', tm = hw_tm_uni_2d_ct, pv = hw_pv_uni_2d_ct)
		pdb.set_trace()
		
		flat_hw_id_tm = []
		for sublist in hw_id_tm_l:
			for item in sublist:
				flat_hw_id_tm.append(item)

		flat_hw_id_pv = []
		for sublist in hw_id_pv_l:
			for item in sublist:
				flat_hw_id_pv.append(item)
		# unique heatwave number
		hw_id_tm_uni = np.unique(np.asarray(flat_hw_id_tm))
		hw_id_pv_uni = np.unique(np.asarray(flat_hw_id_pv))

		if du_wei:
			# if duration weighted
			hw_tm_uni_2d = hw_tm_uni_du_2d
			hw_pv_uni_2d = hw_pv_uni_du_2d
		if ct:
			hw_tm_uni_2d = hw_tm_uni_2d_ct/1000
			hw_pv_uni_2d = hw_pv_uni_2d_ct/1000

	else:
		allyears = False
		if allyears:
			# range0_list = np.arange(-30,1,1)
			# range0_list  = [-15]
			range0_list  = [-7]

			# range0_list  = [-4]
			res_lag = np.zeros([31,3],dtype = np.float16) # the percentage of unique event

			pv_reg_num = [267849,45294,6049]
			tm_reg_num = [116698,4318,34441]
			hw_reg_num = [270376,114794,96906]
			
			row_range0 = [-7,-7,0,0]
			col_range0 = [0,-7,-7,0]
			rownum_list = [7,7,8,8]
			colnum_list = [8,7,7,8]

			hw_blo_pair_tm = np.zeros([3,8],dtype = np.float32)
			hw_blo_pair_pv = np.zeros([3,8],dtype = np.float32)

			hw_blo_hwf_total_tm_pv = np.zeros([3,4],dtype = np.float32) # the heatwave frequency countted by the selected range
			hw_blo_hws_total_tm_pv = np.zeros([3,4],dtype = np.float32) # the heatwave frequency countted by the selected range

			quadrant = False
			for sub_reg in range(3):
				for idx in range(len(range0_list)):
				# for qua in range(4): # for seperate quadranat
					hw_id_tm_l = []
					hw_id_pv_l = []
					if not quadrant:
						range0 = range0_list[idx]
						row_num = (np.abs(range0))*2+1
						row_num = np.int(row_num)
						# row_num = 16 # for e-s and s-e
						col_num = row_num
					else:
						row_num = rownum_list[qua]
						col_num = colnum_list[qua]
				
					hw_tm_lag_2d = np.zeros([row_num,col_num],dtype = np.float16)
					hw_pv_lag_2d = np.zeros([row_num,col_num],dtype = np.float16)

					# unique heatwave number in each grids
					hw_tm_uni_2d = np.zeros([row_num,col_num],dtype = np.float16)
					hw_pv_uni_2d = np.zeros([row_num,col_num],dtype = np.float16)

					# unique heatwave number counting in each grids
					hw_tm_uni_2d_ct = np.zeros([row_num,col_num],dtype = np.float32)
					hw_pv_uni_2d_ct = np.zeros([row_num,col_num],dtype = np.float32)
					
					hw_tm_uni_du_2d = np.zeros([row_num,col_num],dtype = np.float16)
					hw_pv_uni_du_2d = np.zeros([row_num,col_num],dtype = np.float16)

					hw_tm_uni_du_2d_mean = np.zeros([row_num,col_num],dtype = np.float16)
					hw_pv_uni_du_2d_mean = np.zeros([row_num,col_num],dtype = np.float16)


					print(region_name[sub_reg])
					for row in range(row_num):
						print(row)
						for col in range(col_num):
							if not quadrant:
								idx_ss_tm, = np.where((hw_tm_ss < range0 + (row+1)*grid_interval) & (hw_tm_ss >= range0 + (row)*grid_interval) & (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]))
								idx_ee_tm, = np.where((hw_tm_ee < range0+ (col+1)*grid_interval) & (hw_tm_ee >= range0 + (col)*grid_interval)& (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]))

								idx_es_tm, = np.where((hw_tm_es < -3 + (col+1)*grid_interval) & (hw_tm_es >= -3 + (col)*grid_interval)& (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]))
								idx_se_tm, = np.where((hw_tm_se < -12 + (row+1)*grid_interval) & (hw_tm_se >= -12 + (row)*grid_interval)& (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]))


								idx_ss_pv, = np.where((hw_pv_ss < range0+ (row+1)*grid_interval) & (hw_pv_ss >= range0 + (row)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]))
								idx_ee_pv, = np.where((hw_pv_ee < range0 + (col+1)*grid_interval) & (hw_pv_ee >= range0 + (col)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]))

								idx_es_pv, = np.where((hw_pv_es < -3 + (col+1)*grid_interval) & (hw_pv_es >= -3 + (col)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]))
								idx_se_pv, = np.where((hw_pv_se < -12 + (row+1)*grid_interval) & (hw_pv_se >= -12 + (row)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]))

							else:
								idx_ss_tm, = np.where((hw_tm_ss < row_range0[qua] + (row+1)*grid_interval) & (hw_tm_ss >= row_range0[qua] + (row)*grid_interval) & (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]))
								idx_ee_tm, = np.where((hw_tm_ee < col_range0[qua]+ (col+1)*grid_interval) & (hw_tm_ee >= col_range0[qua] + (col)*grid_interval)& (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]))

								# idx_es_tm, = np.where((hw_tm_es < range0 + (col+1)*grid_interval) & (hw_tm_es >= range0 + (col)*grid_interval)& (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]))
								# idx_se_tm, = np.where((hw_tm_se < range0 + (row+1)*grid_interval) & (hw_tm_se >= range0 + (row)*grid_interval)& (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]))


								idx_ss_pv, = np.where((hw_pv_ss < row_range0[qua]+ (row+1)*grid_interval) & (hw_pv_ss >= row_range0[qua] + (row)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]))
								idx_ee_pv, = np.where((hw_pv_ee < col_range0[qua] + (col+1)*grid_interval) & (hw_pv_ee >= col_range0[qua] + (col)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]))

								# idx_es_pv, = np.where((hw_pv_es < range0 + (col+1)*grid_interval) & (hw_pv_es >= range0 + (col)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]))
								# idx_se_pv, = np.where((hw_pv_se < range0 + (row+1)*grid_interval) & (hw_pv_se >= range0 + (row)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]))

							# idx_tm = set(idx_es_tm).intersection(set(idx_se_tm))
							idx_tm = set(idx_ss_tm).intersection(set(idx_ee_tm))
							hw_tm_lag_2d[row,col] = len(idx_tm) # pair num

							idx_tm = [i for i in idx_tm]
							hw_1 =  [idx for idx in hw_id_tm[idx_tm]] # check if necessary ???
							hw_tm_uni_2d[row,col] = len(np.unique(hw_1))/hw_num_sub_region[sub_reg,0] # percentage 
							hw_tm_uni_2d_ct[row,col] = len(np.unique(hw_1))
							hw_id_tm_l.append(hw_1)
							
							hw_du_1 = [idx for idx in hw_du_tm[idx_tm]]
							if len(hw_du_1)>50:
								hw_tm_uni_du_2d_mean[row,col] = np.nanmean(np.asarray(hw_du_1))
							hw_id_du_1 = np.append(np.asarray(hw_1)[:,np.newaxis],np.asarray(hw_du_1)[:,np.newaxis],axis = 1)
							df_hw_1 = pd.DataFrame(hw_id_du_1,columns=['hw_id','hw_du'])
							df2_hw_1 = df_hw_1.groupby('hw_id').apply(lambda x: x['hw_du'].unique())
							if df2_hw_1.empty == False:
								# hw_tm_uni_du_2d[row,col] = df2_hw_1.sum()/hw_num_du_sub_region[sub_reg,0] # percentage of explained frequency 
								hw_tm_uni_du_2d[row,col] = df2_hw_1.sum() # absolute

							# idx_pv = set(idx_es_pv).intersection(set(idx_se_pv)) 
							idx_pv = set(idx_ss_pv).intersection(set(idx_ee_pv)) 
							hw_pv_lag_2d[row,col] = len(idx_pv)
							idx_pv = [i for i in idx_pv]
							hw_2 =  [idx for idx in hw_id_pv[idx_pv]]
							hw_pv_uni_2d[row,col] = len(np.unique(hw_2))/hw_num_sub_region[sub_reg,1]
							hw_pv_uni_2d_ct[row,col] = len(np.unique(hw_2))
							hw_id_pv_l.append(hw_2)

							hw_du_2 = [idx for idx in hw_du_pv[idx_pv]]
							if len(hw_du_2)>50:
								hw_pv_uni_du_2d_mean[row,col] = np.nanmean(np.asarray(hw_du_2))
							hw_id_du_2 = np.append(np.asarray(hw_2)[:,np.newaxis],np.asarray(hw_du_2)[:,np.newaxis],axis = 1)
							df_hw_2 = pd.DataFrame(hw_id_du_2,columns=['hw_id','hw_du'])
							df2_hw_2 = df_hw_2.groupby('hw_id').apply(lambda x: x['hw_du'].unique())
							if df2_hw_2.empty == False:
								# hw_pv_uni_du_2d[row,col] = df2_hw_2.sum()/hw_num_du_sub_region[sub_reg,1]
								hw_pv_uni_du_2d[row,col] = df2_hw_2.sum() # abosolute

					flat_hw_id_tm = []
					for sublist in hw_id_tm_l:
						for item in sublist:
							flat_hw_id_tm.append(item)

					flat_hw_id_pv = []
					for sublist in hw_id_pv_l:
						for item in sublist:
							flat_hw_id_pv.append(item)

					# unique heatwave number accounted by the selected range
					hw_id_tm_uni = np.unique(np.asarray(flat_hw_id_tm))
					hw_id_pv_uni = np.unique(np.asarray(flat_hw_id_pv))

					# hw_blo_pair_tm[sub_reg,qua] = len(flat_hw_id_tm)  # the unique pair
					# hw_blo_pair_pv[sub_reg,qua] = len(flat_hw_id_pv) 
					# # hw_blo_pair_tm[sub_reg,qua+4] = hw_id_tm_uni.shape[0] # the unique heatwave event
					# # hw_blo_pair_pv[sub_reg,qua+4] = hw_id_pv_uni.shape[0] 
					# hw_blo_pair_tm[sub_reg,qua+4] = np.sum(hw_tm_uni_du_2d[np.where(hw_tm_uni_du_2d>0)]/1e3) # the explained hw frequency
					# hw_blo_pair_pv[sub_reg,qua+4] = np.sum(hw_pv_uni_du_2d[np.where(hw_pv_uni_du_2d>0)]/1e3) 

					hw_blo_hwf_total_tm_pv [sub_reg,1] = np.sum(hw_pv_uni_du_2d[np.where(hw_pv_uni_du_2d>0)]/1e3) 
					hw_blo_hwf_total_tm_pv [sub_reg,0] = np.sum(hw_tm_uni_du_2d[np.where(hw_tm_uni_du_2d>0)]/1e3) 
					hw_blo_hwf_total_tm_pv [sub_reg,3] = np.sum(hw_pv_uni_du_2d[np.where(hw_pv_uni_du_2d>0)]/1e3)/(hw_num_du_sub_region[sub_reg,1]/1000)
					hw_blo_hwf_total_tm_pv [sub_reg,2] = np.sum(hw_tm_uni_du_2d[np.where(hw_tm_uni_du_2d>0)]/1e3)/(hw_num_du_sub_region[sub_reg,0]/1000)


					hw_blo_hws_total_tm_pv [sub_reg,1] = hw_id_pv_uni.shape[0]/1e3
					hw_blo_hws_total_tm_pv [sub_reg,0] = hw_id_tm_uni.shape[0]/1e3
					hw_blo_hws_total_tm_pv [sub_reg,3] = (hw_id_pv_uni.shape[0]/1e3)/(hw_num_sub_region[sub_reg,1]/1000)
					hw_blo_hws_total_tm_pv [sub_reg,2] = (hw_id_tm_uni.shape[0]/1e3)/(hw_num_sub_region[sub_reg,0]/1000)

					# np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_' + str(sub_reg) +'_hwd3_ct.npy',hw_tm_uni_2d_ct)
					# np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_' + str(sub_reg) +'_hwd3_ct.npy',hw_pv_uni_2d_ct)

					np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_' + str(sub_reg) +'_hwd3_valid_ct.npy',hw_tm_uni_2d_ct)
					np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_' + str(sub_reg) +'_hwd3_valid_ct.npy',hw_pv_uni_2d_ct)


					# np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_' + str(sub_reg) +'_hwd5_new_ct.npy',hw_tm_uni_2d_ct)
					# np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_' + str(sub_reg) +'_hwd5_new_ct.npy',hw_pv_uni_2d_ct)

					# np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_' + str(sub_reg) +'_blo_new_ct.npy',hw_tm_uni_2d_ct)
					# np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_' + str(sub_reg) +'_blo_new_ct.npy',hw_pv_uni_2d_ct)
		else:
			range0_list  = [-7]
			quadrant = False
			subreg  = False
			if subreg == True:
				for sub_reg in range(3):
					for idx in range(len(range0_list)):
					# for qua in range(4): # for seperate quadranat
						hw_id_tm_l = []
						hw_id_pv_l = []
						if not quadrant:
							range0 = range0_list[idx]
							row_num = (np.abs(range0))*2+1
							row_num = np.int(row_num)
							# row_num = 16 # for e-s and s-e
							col_num = row_num
						else:
							row_num = rownum_list[qua]
							col_num = colnum_list[qua]
					
						hw_tm_lag_2d = np.zeros([row_num,col_num,39],dtype = np.float16)
						hw_pv_lag_2d = np.zeros([row_num,col_num,39],dtype = np.float16)


						print(region_name[sub_reg])
						for row in range(row_num):
							print(row)
							for col in range(col_num):
								for year in range(39):
									idx_ss_tm, = np.where((hw_tm_ss < range0 + (row+1)*grid_interval) & (hw_tm_ss >= range0 + (row)*grid_interval) & (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]) & (hw_year_tm == year))
									idx_ee_tm, = np.where((hw_tm_ee < range0+ (col+1)*grid_interval) & (hw_tm_ee >= range0 + (col)*grid_interval)& (hw_row_tm >= sub_region[sub_reg][0]) & (hw_row_tm < sub_region[sub_reg][-1]) & (hw_year_tm == year))

									idx_ss_pv, = np.where((hw_pv_ss < range0+ (row+1)*grid_interval) & (hw_pv_ss >= range0 + (row)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]) & (hw_year_pv == year))
									idx_ee_pv, = np.where((hw_pv_ee < range0 + (col+1)*grid_interval) & (hw_pv_ee >= range0 + (col)*grid_interval)& (hw_row_pv >= sub_region[sub_reg][0]) & (hw_row_pv < sub_region[sub_reg][-1]) & (hw_year_pv == year))


									# idx_tm = set(idx_es_tm).intersection(set(idx_se_tm))
									idx_tm = set(idx_ss_tm).intersection(set(idx_ee_tm))
									hw_tm_lag_2d[row,col,year] = len(idx_tm) # pair num


									# idx_pv = set(idx_es_pv).intersection(set(idx_se_pv)) 
									idx_pv = set(idx_ss_pv).intersection(set(idx_ee_pv)) 
									hw_pv_lag_2d[row,col,year] = len(idx_pv)


						np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_' + str(sub_reg) +'_hwd3_valid_each_year_ct.npy',hw_tm_lag_2d)
						np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_' + str(sub_reg) +'_hwd3_valid_each_year_ct.npy',hw_pv_lag_2d)

			else:
				for idx in range(len(range0_list)):
				# for qua in range(4): # for seperate quadranat
					hw_id_tm_l = []
					hw_id_pv_l = []
					if not quadrant:
						range0 = range0_list[idx]
						row_num = (np.abs(range0))*2+1
						row_num = np.int(row_num)
						# row_num = 16 # for e-s and s-e
						col_num = row_num
					else:
						row_num = rownum_list[qua]
						col_num = colnum_list[qua]
				
					hw_tm_lag_2d = np.zeros([row_num,col_num,39],dtype = np.float16)
					hw_pv_lag_2d = np.zeros([row_num,col_num,39],dtype = np.float16)

					for row in range(row_num):
						print(row)
						for col in range(col_num):
							for year in range(39):
								idx_ss_tm, = np.where((hw_tm_ss < range0 + (row+1)*grid_interval) & (hw_tm_ss >= range0 + (row)*grid_interval)  & (hw_year_tm == year))
								idx_ee_tm, = np.where((hw_tm_ee < range0+ (col+1)*grid_interval) & (hw_tm_ee >= range0 + (col)*grid_interval) & (hw_year_tm == year))

								idx_ss_pv, = np.where((hw_pv_ss < range0+ (row+1)*grid_interval) & (hw_pv_ss >= range0 + (row)*grid_interval) & (hw_year_pv == year))
								idx_ee_pv, = np.where((hw_pv_ee < range0 + (col+1)*grid_interval) & (hw_pv_ee >= range0 + (col)*grid_interval) & (hw_year_pv == year))


								# idx_tm = set(idx_es_tm).intersection(set(idx_se_tm))
								idx_tm = set(idx_ss_tm).intersection(set(idx_ee_tm))
								hw_tm_lag_2d[row,col,year] = len(idx_tm) # pair num


								# idx_pv = set(idx_es_pv).intersection(set(idx_se_pv)) 
								idx_pv = set(idx_ss_pv).intersection(set(idx_ee_pv)) 
								hw_pv_lag_2d[row,col,year] = len(idx_pv)


					np.save(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_whole_region_hwd3_valid_each_year_ct.npy',hw_tm_lag_2d)
					np.save(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_whole_region_hwd3_valid_each_year_ct.npy',hw_pv_lag_2d)




		
	''' initial submisson'''
	# sio.savemat(file_dir + '/heatwave_event/hw_blo_grid_ev/hw_pv_lag_unique_hwf_explained_by_range7_' + str(sub_reg) +'.mat',{'hwf':hw_blo_hwf_total_tm_pv})

	''' first revision '''
	# np.savez(file_dir + '/cpc_tmp/hws_bls_lag_unique_hwf_explained_by_range7_hwd3_1202.npz', hwf= hw_blo_hwf_total_tm_pv,hws= hw_blo_hws_total_tm_pv)
	# np.savez(file_dir + '/cpc_tmp/hws_bls_lag_unique_hwf_explained_by_range7_blo_1202.npz', hwf= hw_blo_hwf_total_tm_pv,hws= hw_blo_hws_total_tm_pv)
	# np.savez(file_dir + '/cpc_tmp/hws_bls_lag_unique_hwf_explained_by_range4_hwd3.npz', hwf= hw_blo_hwf_total_tm_pv,hws= hw_blo_hws_total_tm_pv)
	# np.savez(file_dir + '/cpc_tmp/hws_bls_lag_unique_hwf_explained_by_range4_blo.npz', hwf= hw_blo_hwf_total_tm_pv,hws= hw_blo_hws_total_tm_pv)

	# np.savez(file_dir + '/cpc_tmp/hws_bls_lag_unique_hwf_explained_by_range7_hwd3_valid.npz', hwf= hw_blo_hwf_total_tm_pv,hws= hw_blo_hws_total_tm_pv)
	pdb.set_trace()


def asso_var():
	prob = True
	bl_in_hw = True
	whole_region = True
	file_dir = '/home/user/Documents/research/project1'
	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()

	# ta_90th_3d = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_ori']
	ta_90th_3d = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_detrend']
	
	ta_90th = np.transpose(ta_90th_3d,[2,0,1])
	ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')
	ta_90th = time_filter_new(ta_90th,3)


	hw_tm = co_occur(ta_90th,blo_sum_tm90,False,False)
	hw_pv = co_occur(ta_90th,blo_sum_weak,False,False)


	sub_reg = [[0,40],[40,60],[60,80]]


	if whole_region:
		hw_bl_var = np.zeros([39,2],dtype = np.float16) 
	else:
		hw_bl_var = np.zeros([39,2,3],dtype = np.float16) 

	for day in range(hw_tm.shape[0]):
		for year in range(hw_tm.shape[1]):
			hw_tm[day,year,:,:][np.where(land_mask == 0 )] = np.nan
			hw_pv[day,year,:,:][np.where(land_mask == 0 )] = np.nan

	for year in range(39):
		hw_tm_year = hw_tm[:,year,:,:]
		hw_pv_year = hw_pv[:,year,:,:]

		if whole_region:
			hw_bl_var[year,0] = np.nansum(hw_tm_year[~np.isnan(hw_tm_year)])
			hw_bl_var[year,1] = np.nansum(hw_pv_year[~np.isnan(hw_pv_year)])
		else:
			for reg in range(3):
				hw_tm_year_reg = hw_tm_year[:,sub_reg[reg][0]:sub_reg[reg][1],:]
				hw_pv_year_reg = hw_pv_year[:,sub_reg[reg][0]:sub_reg[reg][1],:]
				
				hw_bl_var[year,0,reg] = np.nansum(hw_tm_year_reg[~np.isnan(hw_tm_year_reg)])
				hw_bl_var[year,1,reg] = np.nansum(hw_pv_year_reg[~np.isnan(hw_pv_year_reg)])


		if prob:
			tm = blo_sum_tm90[:,year,:,:]
			pv = blo_sum_weak[:,year,:,:]
			hw =  ta_90th[:,year,:,:]
			for day in range(92):
				tm[day,:,:][np.where(land_mask == 0 )] = np.nan
				pv[day,:,:][np.where(land_mask == 0 )] = np.nan
				hw[day,:,:][np.where(land_mask == 0 )] = np.nan
			
			if whole_region:
				tm_year = np.nansum(tm[~np.isnan(tm)])
				pv_year = np.nansum(pv[~np.isnan(pv)])
				hw_year = np.nansum(hw[~np.isnan(hw)])			

				if bl_in_hw:
					hw_bl_var[year,0] = hw_bl_var[year,0]/hw_year
					hw_bl_var[year,1] = hw_bl_var[year,1]/hw_year
				else:
					hw_bl_var[year,0] = hw_bl_var[year,0]/tm_year
					hw_bl_var[year,1] = hw_bl_var[year,1]/pv_year	


			else:
				for reg in range(3):
					tm_year_reg = tm[:,sub_reg[reg][0]:sub_reg[reg][1],:]
					pv_year_reg = pv[:,sub_reg[reg][0]:sub_reg[reg][1],:]
					
					tm_year_reg = np.nansum(tm_year_reg[~np.isnan(tm_year_reg)])
					pv_year_reg = np.nansum(pv_year_reg[~np.isnan(pv_year_reg)])

					hw_bl_var[year,0,reg] = hw_bl_var[year,0,reg]/tm_year_reg
					hw_bl_var[year,1,reg] = hw_bl_var[year,1,reg]/pv_year_reg


	# np.save(file_dir + '/cpc_tmp/hw_bl_concurrence_detrend_prob_subreg.npy',hw_bl_var)
	np.save(file_dir + '/cpc_tmp/hw_bl_concurrence_detrend_prob_bl_in_hw.npy',hw_bl_var)
	# np.save(file_dir + '/cpc_tmp/bl_num.npy',hw_bl_var)
	pdb.set_trace()


# main_menclo_test()
main_lag_ev_based()
# asso_var()