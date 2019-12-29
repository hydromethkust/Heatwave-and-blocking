import numpy as np 
import scipy.io as sio
from PIL import Image
import pdb
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import tifffile as tiff
from plot_code import geo_grid, geo_grid_2,plot_bar,geo_plot_point,plot_cdf,tsplot,plot_grid

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
import rpy2
# pdb.set_trace()
from hw_blo_basic import time_filter_new, time_filter
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects.numpy2ri as rpyn
from  pyKstest import kstest2 
from operator import itemgetter
import itertools
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from hw_blo_basic import load_data,mon_blo_freq, make_ax, daynum, gridnum, daily_gridnum, persistence, block_freq, co_occur, condi_prob, cor_relation_2d, cor_relation_1d, geomap_china,lon_mean_blocking,lag_co_occur,event_number


# figure1-1: hw spatial distribution, hw sequence distribution, trend, trend
# figure 1-2: hw trend -- line plot
# figure 1-3: hw/blo  grid number
# figure 2-1: BLF BLS 
# figure 2-2: conditionl probability 
# figure 3 : temporal association
# figure 4 : kstest

def hwf_hws():
	''' spatial distribution of hw frequency and hw sequence '''
	two_figure = True
	file_dir = '/home/user/Documents/research/project1'
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
	pv_label = sio.loadmat(file_dir +'/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.4_daily_extent_100_with_label.mat')['blocking']
	
	# ta_90th = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N_detrend_105_39_sig_only.mat')['hw']
	# ta_90th = sio.loadmat(file_dir + '/cpc_tmp/hw_cpc_3d.mat')['hw_detrend']
	ta_90th = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_ori']
	# pdb.set_trace()
	ta_90th = np.transpose(ta_90th,[2,0,1])
	ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')
	hw_sum = ta_90th 

	''' hwf blf''' 	
	# blo_sum_tm90 = time_filter_new (blo_sum_tm90,5) # begin and end days may lack 5 days
	blo_tm_freq = block_freq(blo_sum_tm90,dimen = 'grid')/92/39
	blo_tm_freq[np.where(land_mask==0)] = np.nan
	
	blo_pv_freq = block_freq(blo_sum_weak,dimen = 'grid')/92/39
	blo_pv_freq[np.where(land_mask==0)] = np.nan

	outfig2_1 = file_dir + '/paper_figure/figure2_1_blf.pdf'
	single_data = False
	# title1  = '(a) P(blocking)'
	# title2  = '(b) P(blocking)'
	title1  = '(a) Blocking frequency [TM index]'
	title2  = '(b) Blocking frequency [PV index]'
	clevs = np.arange(0,0.11,0.01)
	
	# geomap_china(blo_tm_freq, blo_pv_freq, clevs,clevs,title1,title2,outfig2_1,single_data) 


	hw_sum_freq = block_freq(hw_sum,dimen = 'grid')/92/39
	hw_sum_freq[np.where(land_mask==0)] = np.nan

	''' hws bls '''
	hw_ev_num, hw_ev_du,hw_num_year = event_number(hw_sum, hw = True, tm = False, pv = False)
	tm_ev_num, tm_ev_du,tm_num_year = event_number(blo_sum_tm90, hw = False, tm = True, pv = False)
	pv_ev_num, pv_ev_du,pv_num_year = event_number(pv_label, hw = False, tm = False, pv = True)
	
	hw_ev_num[np.where(land_mask==0 )] = np.nan
	hw_ev_du[np.where(land_mask==0 )] = np.nan
	tm_ev_num[np.where(land_mask==0 )] = np.nan
	tm_ev_du[np.where(land_mask==0 )] = np.nan
	pv_ev_num[np.where(land_mask==0 )] = np.nan
	pv_ev_du[np.where(land_mask==0 )] = np.nan

	outfig1_1 = file_dir + '/paper_figure/figure1_1_hwf_cpc.pdf'
	outfig1_1_1 = file_dir + '/paper_figure/figure1_1_hws_cpc.pdf'
	single_data = True
	title1  = '(a) Heatwave frequency'
	title2  = '(b) Heatwave sequence'
	clevs1 = np.arange(0,0.11,0.01)
	clevs2 = np.arange(0,2.2,0.2)
	# geomap_china(hw_sum_freq, hw_sum_freq, clevs1,clevs1,title1,title1,outfig1_1,single_data)
	# geomap_china(hw_ev_num/39, hw_ev_num/39, clevs2,clevs2,title2,title2,outfig1_1_1,single_data)


	outfig2_2 = file_dir + '/paper_figure/figure2_2_tms_pvs.pdf'
	single_data = False
	# title1  = '(c) Blocking sequence'
	# title2  = '(d) Blocking sequence'
	title1  = '(c) Blocking sequence [TM index]'
	title2  = '(d) Blocking sequence [PV index]'
	clevs2 = np.arange(0,2.2,0.2)
	geomap_china(tm_ev_num/39, pv_ev_num/39, clevs2,clevs2,title1,title2,outfig2_2,single_data)
	pdb.set_trace()


	''' spatial trend of hwf and hws '''
	hwf = np.sum(hw_sum,axis=0)
	r_grid_hwf = np.zeros([hwf.shape[1],hwf.shape[2],2],dtype = np.float16)
	r_grid_hws = np.zeros([hwf.shape[1],hwf.shape[2],2],dtype = np.float16)
	for row in range(hwf.shape[1]):
		for col in range(hwf.shape[2]):
				x_data = range(1979,2018)
				r_grid_hwf[row,col,0],intercept, r, r_grid_hwf[row,col,1], std_err=linregress(x_data,np.squeeze(hwf[:,row,col]))
				r_grid_hws[row,col,0],intercept, r, r_grid_hws[row,col,1], std_err=linregress(x_data,np.squeeze(hw_num_year[row,col,:]))

	# r_grid_copy = r_grid.copy()
	r_grid_hwf[np.where(land_mask == 0)] = np.nan
	r_grid_hws[np.where(land_mask == 0)] = np.nan


	r_grid_hwf[:,:,0][np.where(r_grid_hwf[:,:,1]>0.05)] = 0
	r_grid_hws[:,:,0][np.where(r_grid_hws[:,:,1]>0.05)] = 0

	r_hwf_nan = np.where(~np.isnan(r_grid_hwf))
	r_hws_nan = np.where(~np.isnan(r_grid_hwf))

	r_hwf_in = np.where(r_grid_hwf > 0.15)
	r_hws_in = np.where(r_grid_hwf > 0.035)


	r_hwf_mean = np.nanmean(hw_sum_freq[~np.isnan(hw_sum_freq)])
	r_hws_mean = np.nanmean(hw_ev_num[~np.isnan(hw_ev_num)])/39
	pdb.set_trace()


	# idx_nan = np.isnan(r_grid_copy[:,:,0])
	# idx_v = np.where(r_grid_copy[:,:,0]>0.035)
	# print(idx_v[0].shape[0]/(80*180-np.sum(idx_nan)))

	clevs2 = np.arange(-0.1,0.12,0.02)
	clevs1 = np.arange(-0.5,0.6,0.1)
	
	title1 = '(c) Trend of heatwave frequency'
	title2 = '(d) Trend of heatwave sequence'

	outfig1_2 = file_dir +'/paper_figure/figrure1_2_trend_of_hwf_cpc.pdf'
	outfig1_3 = file_dir +'/paper_figure/figrure1_3_trend_of_hws_cpc.pdf'

	single_data = False
	geomap_china(np.squeeze(r_grid_hwf[:,:,0]),np.squeeze(r_grid_hwf[:,:,1]), clevs1,clevs1,title1,title1,outfig1_2, single_data,sig=True)
	geomap_china(np.squeeze(r_grid_hws[:,:,0]),np.squeeze(r_grid_hws[:,:,1]), clevs2,clevs2,title2,title2,outfig1_3, single_data,sig=True)


def hw_trend_line():
	plot_2darr = True
	fsize = 13
	file_dir = '/home/user/Documents/research/project1'

	if plot_2darr:
		# hw_daynum = sio.loadmat(file_dir + '/temporal_result/trend/ratio_thres_hw_90th_3d_blo_pv_1.0_0.35_extent_4000_15_5_valid_nondetrend.mat')['ratio_hw_day']
		# hw_gridnum = sio.loadmat(file_dir + '/temporal_result/trend/ratio_thres_hw_90th_3d_blo_pv_1.0_0.35_extent_4000_15_5_valid_nondetrend.mat')['ratio_hw_grid']
		# hw_len = sio.loadmat(file_dir + '/heatwave_event/hw_fea_trend_ratio_nondetrend.mat')['len']
		# hw_area = sio.loadmat(file_dir + '/heatwave_event/hw_fea_trend_ratio_nondetrend.mat')['area']
		# hw_extent = sio.loadmat(file_dir + '/heatwave_event/hw_fea_trend_ratio_nondetrend.mat')['extent']
		# hw_intensity = sio.loadmat(file_dir + '/heatwave_event/hw_fea_trend_ratio_nondetrend.mat')['intes']

		# revised with pv 1.2 0.4 hw 0.4
		''' initial submission '''
		# hw_daynum = sio.loadmat(file_dir + '/temporal_result/trend/ratio_thres_hw_90th_3d_blo_pv_1.2_0.4_extent_4000_15_5_valid_nondetrend.mat')['ratio_hw_day']
		# hw_gridnum = sio.loadmat(file_dir + '/temporal_result/trend/ratio_thres_hw_90th_3d_blo_pv_1.2_0.4_extent_4000_15_5_valid_nondetrend.mat')['ratio_hw_grid']

		# hw_len = sio.loadmat(file_dir + '/heatwave_event/hw_fea_trend_ratio_new_pv_1.2_0.4_hw_0.4.mat')['len']
		# hw_area = sio.loadmat(file_dir + '/heatwave_event/hw_fea_trend_ratio_new_pv_1.2_0.4_hw_0.4.mat')['area']
		# hw_extent = sio.loadmat(file_dir + '/heatwave_event/hw_fea_trend_ratio_new_pv_1.2_0.4_hw_0.4.mat')['extent']
		# hw_intensity = sio.loadmat(file_dir + '/heatwave_event/hw_fea_trend_ratio_new_pv_1.2_0.4_hw_0.4.mat')['intes']
		# pdb.set_trace()

		''' first major revision '''

		hw_daynum = np.load(file_dir + '/cpc_tmp/ratio_thres_hw_90th_3d_blo_pv_1.2_0.4_extent_4000_15_5_valid_nondetrend.npz')['ratio_hw_day']
		hw_gridnum = np.load(file_dir + '/cpc_tmp/ratio_thres_hw_90th_3d_blo_pv_1.2_0.4_extent_4000_15_5_valid_nondetrend.npz')['ratio_hw_grid']

		hw_len = np.load(file_dir + '/cpc_tmp/hw_fea_trend_ratio_new_pv_1.2_0.4_hw_0.4.npz')['len']
		hw_area = np.load(file_dir + '/cpc_tmp/hw_fea_trend_ratio_new_pv_1.2_0.4_hw_0.4.npz')['area']
		hw_extent = np.load(file_dir + '/cpc_tmp/hw_fea_trend_ratio_new_pv_1.2_0.4_hw_0.4.npz')['extent']
		hw_intensity = np.load(file_dir + '/cpc_tmp/hw_fea_trend_ratio_tm90_hw_0.4_nondetrend_new.npz')['intes']
		pdb.set_trace()




		len_thres = [0,3,4,5,6]
		extent_thres = [0,30,100,300,1000]
		area_thres = [0,100,300,1000,3000]
		intensity_thres = [0,100,300,1000,3000]
		# intensity_thres = [0,100,300,500,3000]

		x = range(1979,2018)
		# color = ['k','g','b','y','r']
		# color = ['k','k','b','r','g']
		# color = ['k','grey','coral','cornflowerblue','g']
		# color = ['k', sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"],sns.xkcd_rgb["denim blue"],'g']
		# color = ['k', '#4575b4', '#fc8d59','#af8dc3','g']
		# color = ['k', '#8dd3c7', '#fc8d62','#bebada','g']
		color = ['k', '#4d4d4d', '#4575b4','#ef8a62','g']
		
		# color = ['k', sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"],sns.xkcd_rgb["dark blue grey"],'g']
		# color= ['g','coral','cornflowerblue','dimgrey','y']
		outfig1 = file_dir + ('/paper_figure/cpc_hw_daynum_trend_new.pdf')
		outfig2 = file_dir + ('/paper_figure/cpc_hw_gridnum_trend_new.pdf')
		outfig3 = file_dir + ('/paper_figure/cpc_hw_len_trend_new.pdf')
		outfig4 = file_dir + ('/paper_figure/cpc_hw_area_trend_new.pdf')
		outfig5 = file_dir + ('/paper_figure/cpc_hw_extent_trend_new.pdf')
		# outfig6 = file_dir + ('/paper_figure/cpc_hw_intens_trend_new.pdf')
		outfig6 = file_dir + ('/paper_figure/cpc_hw_intens_trend_intens_correct.pdf')
		
		# fig_day = plt.figure(figsize=(12,8))
		fig_day = plt.figure(figsize=(6,4))
		plt.rcParams["font.family"] = "serif"
		# ax = fig_day.add_axes()
		for i in range(1,4):
			plt.plot(x,np.squeeze(hw_daynum[:,i]),color= color[i],linestyle = '-',marker ='.',label = 'grid_thres_' + str(i*160),linewidth= 1.5) # 0 or 160???
			# pdb.set_trace()
			z = np.polyfit(x,np.squeeze(hw_daynum[:,i]).astype(np.float32),1)
			p = np.poly1d(z)
			# plt.plot(x,p(x),color[i]+'--',linewidth= 1.5)
			plt.plot(x,p(x),color = color[i],linestyle = '--',linewidth= 1.5)

		# plt.title('Daynum ratio',fontsize = 12)
		plt.title('(e) Heatwave days',fontsize = fsize)
		# plt.legend(['grid_thres_80', 'grid_thres_160','grid_thres_240'])
		# plt.legend()
		plt.legend(prop={'size': 11})
		plt.xlabel('Year',fontsize = fsize)
		plt.ylabel('Ratio',fontsize = fsize)
		plt.xlabel('Year',fontsize = fsize)
		plt.ylabel('Ratio',fontsize = fsize)
		# ax.set_xticks(fontsize = 14)
		# ax.set_yticks(fontsize = 14)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.tight_layout()
		# fig_day.savefig(outfig1)
		plt.show()


		# fig_grid = plt.figure(figsize=(12,8))
		fig_grid = plt.figure(figsize=(6,4))
		plt.rcParams["font.family"] = "serif"
		# ax = fig_grid.add_axes()
		for i in range(1,4):
			# plt.plot(x,np.squeeze(hw_gridnum[:,i]),color[i]+ '.-', label = 'day_thres'+ str((3+(i-1)*2)),linewidth= 1.5)
			plt.plot(x,np.squeeze(hw_gridnum[:,i]),color= color[i],linestyle = '-',marker ='.', label = 'day_thres_'+ str(3*i),linewidth= 1.5)
			z = np.polyfit(x,np.squeeze(hw_daynum[:,i]).astype(np.float32),1)
			p = np.poly1d(z)
			plt.plot(x,p(x),color = color[i],linestyle = '--',linewidth= 1.5)
			# plt.plot(x,p(x),color[i]+'--',linewidth= 1.5)
		# plt.title('Gridnum ratio',fontsize = 12)
		plt.title('(f) Heatwave grids',fontsize = fsize)
		plt.legend(prop={'size': 11})
		plt.xlabel('Year',fontsize = fsize)
		plt.ylabel('Ratio',fontsize = fsize)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		# ax.set_xticks(fontsize = 14)
		# ax.set_yticks(fontsize = 14)
		plt.tight_layout()
		# fig_grid.savefig(outfig2)
		plt.show()
		pdb.set_trace()

		fig_len = plt.figure(figsize=(6,4))
		plt.rcParams["font.family"] = "serif"
		# ax = fig_len.add_axes()
		for i in range(1,4):
			plt.plot(x,np.squeeze(hw_len[:,i]),color= color[i],linestyle = '-',marker ='.', label = 'HWD_thres_'+ str(len_thres[i]),linewidth= 1.5)
			# plt.plot(x,np.squeeze(hw_len[:,i]),color[i]+'.-', label = 'len_' + str(len_thres[i]),linewidth= 1.5)
			z = np.polyfit(x,np.squeeze(hw_daynum[:,i]).astype(np.float32),1)
			p = np.poly1d(z)
			# plt.plot(x,p(x),color[i]+'--',linewidth= 1.5)
			plt.plot(x,p(x),color = color[i],linestyle = '--',linewidth= 1.5)
		plt.title('(g) Event number (HWD)',fontsize = fsize)
		plt.legend(prop={'size': 11})
		plt.xlabel('Year',fontsize = fsize)
		plt.ylabel('Ratio',fontsize = fsize)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		# ax.set_xticks(fontsize = 14)
		# ax.set_yticks(fontsize = 14)
		plt.tight_layout()
		# fig_len.savefig(outfig3)
		plt.show()

		fig_area = plt.figure(figsize=(6,4))
		plt.rcParams["font.family"] = "serif"
		# ax = fig_area.add_axes()
		for i in range(1,4):
			plt.plot(x,np.squeeze(hw_area[:,i]),color= color[i],linestyle = '-',marker ='.', label = 'HWV_thres_'+ str(area_thres[i]),linewidth= 1.5)
			# plt.plot(x,np.squeeze(hw_area[:,i]),color[i]+'.-', label = 'area_' + str(area_thres[i]),linewidth= 1.5)
			z = np.polyfit(x,np.squeeze(hw_daynum[:,i]).astype(np.float32),1)
			p = np.poly1d(z)
			# plt.plot(x,p(x),color[i]+'--',linewidth= 1.5)
			plt.plot(x,p(x),color = color[i],linestyle = '--',linewidth= 1.5)
		plt.title('(h) Event number (HWV)',fontsize = fsize)
		plt.legend(prop={'size': 11})
		plt.xlabel('Year',fontsize = fsize)
		plt.ylabel('Ratio',fontsize = fsize)
		# ax.set_xticks(fontsize = 14)
		# ax.set_yticks(fontsize = 14)
		plt.tight_layout()
		# fig_area.savefig(outfig4)
		plt.show()

		fig_extent = plt.figure(figsize=(6,4))
		plt.rcParams["font.family"] = "serif"
		for i in range(1,4):
			plt.plot(x,np.squeeze(hw_extent[:,i]),color= color[i],linestyle = '-',marker ='.', label = 'HWE_thres_'+ str(extent_thres[i]),linewidth= 1.5)
			# plt.plot(x,np.squeeze(hw_extent[:,i]),color[i]+'.-', label = 'extent_' + str(extent_thres[i]),linewidth= 1.5)
			z = np.polyfit(x,np.squeeze(hw_daynum[:,i]).astype(np.float32),1)
			p = np.poly1d(z)
			# plt.plot(x,p(x),color[i]+'--',linewidth= 1.5)
			plt.plot(x,p(x),color = color[i],linestyle = '--',linewidth= 1.5)
		plt.title('(i) Event number (HWE)',fontsize = fsize)
		plt.legend(prop={'size': 11})
		plt.xlabel('Year',fontsize = fsize)
		plt.ylabel('Ratio',fontsize = fsize)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		# ax.set_xticks(fontsize = 14)
		# ax.set_yticks(fontsize = 14)
		plt.tight_layout()
		# fig_extent.savefig(outfig5)
		plt.show()


		fig_intens = plt.figure(figsize=(6,4))
		plt.rcParams["font.family"] = "serif"
		for i in range(1,4):
			plt.plot(x,np.squeeze(hw_intensity[:,i]),color= color[i],linestyle = '-',marker ='.', label = 'HWI_thres_'+ str(intensity_thres[i]),linewidth= 1.5)
			# plt.plot(x,np.squeeze(hw_intensity[:,i]),color[i]+'.-', label = 'intensity_' + str(intensity_thres[i]),linewidth= 1.5)
			z = np.polyfit(x,np.squeeze(hw_daynum[:,i]).astype(np.float32),1)
			p = np.poly1d(z)
			# plt.plot(x,p(x),color[i]+'--',linewidth= 1.5)
			plt.plot(x,p(x),color = color[i],linestyle = '--',linewidth= 1.5)
		plt.title('(j) Event number (HWI)',fontsize = fsize)
		plt.legend(prop={'size': 11})
		plt.xlabel('Year',fontsize = fsize)
		plt.ylabel('Ratio',fontsize = fsize)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		# ax.set_xticks(fontsize = 14)
		# ax.set_yticks(fontsize = 14)
		plt.tight_layout()
		fig_intens.savefig(outfig6)
		plt.show()

	else:
		x = range(1979,2018)
		# prob_tm90 = sio.loadmat(file_dir + '/temporal_result/trend/blo_tm90_related_hw_nondetrend_yearly_prob_trend.mat')['prob']
		# prob_pv = sio.loadmat(file_dir + '/temporal_result/trend/blo_pv_1.0_0.35_related_hw_nondetrend_yearly_prob_trend.mat')['prob']
		prob_tm90 = sio.loadmat(file_dir + '/temporal_result/trend/blo_tm90_related_hw_detrend_sig_yearly_prob_trend.mat')['prob']
		prob_pv = sio.loadmat(file_dir + '/temporal_result/trend/blo_pv_1.0_0.35_related_hw_detrend_sig_yearly_prob_trend.mat')['prob']

		fig_prob = plt.figure(figsize=(12,8))
		plt.plot(x,prob_tm90,'k.-', label = 'TM90',linewidth= 1.5)
		z = np.polyfit(x,np.squeeze(prob_tm90),1)
		p = np.poly1d(z)
		plt.plot(x,p(x),'k--',linewidth= 1.5)

		plt.plot(x,prob_pv,'r.-', label = 'PV_1.0',linewidth= 1.5)
		z = np.polyfit(x,np.squeeze(prob_pv),1)
		p = np.poly1d(z)
		plt.plot(x,p(x),'r--',linewidth= 1.5)


		
		outfig = file_dir + ('/result_figure/hw/hw_detrend_condi_prob_based_blo.png')
		plt.title('Conditional probability of detrend heatwave based on blocking',fontsize = 16)
		plt.legend()
		plt.xlabel('Year',fontsize = 14)
		plt.ylabel('Probability',fontsize = 14)
		plt.tight_layout()
		fig_prob.savefig(outfig)
		plt.show()



def hw_bl_var_line():
	''''first inition, variability of concurrent association '''
	file_dir = '/home/user/Documents/research/project1'
	# ori = np.load(file_dir + '/cpc_tmp/hw_bl_concurrence_nondetrend.npy')
	# detrend = np.load(file_dir + '/cpc_tmp/hw_bl_concurrence.npy')

	ori = np.load(file_dir + '/cpc_tmp/hw_bl_concurrence_nondetrend_prob.npy')
	detrend = np.load(file_dir + '/cpc_tmp/hw_bl_concurrence_detrend_prob.npy')

	# ori = np.load(file_dir + '/cpc_tmp/hw_bl_concurrence_nondetrend_prob_bl_in_hw.npy')
	# detrend = np.load(file_dir + '/cpc_tmp/hw_bl_concurrence_detrend_prob_bl_in_hw.npy')

	# ori = np.load(file_dir + '/cpc_tmp/hw_bl_concurrence_nondetrend_prob_subreg.npy')
	# detrend = np.load(file_dir + '/cpc_tmp/hw_bl_concurrence_detrend_prob_subreg.npy')

	# ori = ori[:,:,1]
	# detrend = detrend[:,:,1]
	# pdb.set_trace()

	# ori[np.isnan(ori)] = 0
	# detrend[np.isnan(detrend)] = 0

	# ori = np.load(file_dir + '/cpc_tmp/bl_num.npy')
	# detrend = np.load(file_dir + '/cpc_tmp/bl_num.npy')
	

	color = ['#4575b4','#f19774']
	fsize = 12

	fig = plt.figure(figsize=(12,4))
	plt.rcParams["font.family"] = "serif"
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
	
	ax0 = plt.subplot(gs[0])
	x = range(1979,2018)
	plt.plot(x,np.squeeze(detrend[:,0]),color= color[0],linestyle = '-',marker ='.', label = 'After detrending',linewidth= 1.5)
	plt.plot(x,np.squeeze(ori[:,0]),color= color[1],linestyle = '-',marker ='.', label = 'Before detrending',linewidth= 1.5)

	z0 = np.polyfit(x,np.squeeze(detrend[:,0]).astype(np.float32),1)
	p0 = np.poly1d(z0)

	z1 = np.polyfit(x,np.squeeze(ori[:,0]).astype(np.float32),1)
	p1 = np.poly1d(z1)
	# plt.plot(x,p(x),color[i]+'--',linewidth= 1.5)
	plt.plot(x,p0(x),color = color[0],linestyle = '--',linewidth= 1.5)
	plt.plot(x,p1(x),color = color[1],linestyle = '--',linewidth= 1.5)

	# plt.title('Concurrence number(HW-TM)',fontsize = fsize)
	# plt.title('Concurrence probability(HW-TM,North)',fontsize = fsize)
	# plt.title('P(heatwave|blocking)(HW-TM,Middle)',fontsize = fsize)

	plt.title('(a) P(heatwave|blocking) [TM index]',fontsize = fsize)
	# plt.title('(c) P(blocking|heatwave) [TM index]',fontsize = fsize)
	plt.legend(prop={'size': 11})
	plt.xlabel('Year',fontsize = fsize)
	# plt.ylabel('Number',fontsize = fsize)
	plt.ylabel('Probability',fontsize = fsize)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.ylim([0,0.55])
	# plt.ylim([0,22000])


	ax1 = plt.subplot(gs[1])
	x = range(1979,2018)
	plt.plot(x,np.squeeze(detrend[:,1]),color= color[0],linestyle = '-',marker ='.', label = 'After detrending',linewidth= 1.5)
	plt.plot(x,np.squeeze(ori[:,1]),color= color[1],linestyle = '-',marker ='.', label = 'Before detrending',linewidth= 1.5)

	z0 = np.polyfit(x,np.squeeze(detrend[:,1]).astype(np.float32),1)
	p0 = np.poly1d(z0)

	z1 = np.polyfit(x,np.squeeze(ori[:,1]).astype(np.float32),1)
	p1 = np.poly1d(z1)
	# plt.plot(x,p(x),color[i]+'--',linewidth= 1.5)
	plt.plot(x,p0(x),color = color[0],linestyle = '--',linewidth= 1.5)
	plt.plot(x,p1(x),color = color[1],linestyle = '--',linewidth= 1.5)

	# plt.title('Concurrence number (HW-PV)',fontsize = fsize)
	# plt.title('Concurrence probabity (HW-PV,North)',fontsize = fsize)
	# plt.title('P(heatwave|blocking) (HW-PV,Middle)',fontsize = fsize)
	plt.title('(b) P(heatwave|blocking) [PV index]',fontsize = fsize)
	# plt.title('(d) P(blocking|heatwave) [PV index]',fontsize = fsize)
	plt.legend(prop={'size': 11})
	plt.xlabel('Year',fontsize = fsize)
	# plt.ylabel('Number',fontsize = fsize)
	plt.ylabel('Probability',fontsize = fsize)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.ylim([0,0.55])
	# plt.ylim([0,22000])
	
	
	plt.tight_layout()
	plt.show()
	# pdb.set_trace()
	fig.savefig(file_dir + '/paper_figure/cpc_concurrence_prob_whole_hw_in_bl_revise.pdf')

def  tsplot_lag():
	'''' ts plot for extended temopral association with time lag for each year '''
	file_dir = '/home/user/Documents/research/project1'
	tm_lag  = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_whole_region_hwd3_valid_each_year_ct.npy') 
	pv_lag  = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_whole_region_hwd3_valid_each_year_ct.npy')

	tm_lag_ss = np.squeeze(np.nansum(tm_lag,axis = 1))
	tm_lag_ee = np.squeeze(np.nansum(tm_lag,axis = 0))

	pv_lag_ss = np.squeeze(np.nansum(pv_lag,axis = 1))
	pv_lag_ee = np.squeeze(np.nansum(pv_lag,axis = 0))
	# pdb.set_trace()

	tm_ss_mean = np.nanmean(tm_lag_ss,axis = 1)
	tm_ee_mean = np.nanmean(tm_lag_ee,axis = 1)
	pv_ss_mean = np.nanmean(pv_lag_ss,axis = 1)
	pv_ee_mean = np.nanmean(pv_lag_ee,axis = 1)


	lag_range = np.arange(-7,8,1)
	tsplot_range = False

	if tsplot_range:
		fb_fig = plt.figure(figsize=(6, 5))
		plt.rcParams["font.family"] = "serif"
		gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
		ax0 = plt.subplot(gs[0])
		tsplot(tm_lag_ss,lag_range,color= 'coral',line_color= 'orangered',label= 'Gap_ss [TM index]',plot_mean = True,plot_median=False)
		tsplot(tm_lag_ee,lag_range,color= 'dodgerblue',line_color= 'blue',label ='Gap_ee [TM index]',plot_mean = True,plot_median=False)
		ax0.yaxis.tick_right()
		plt.ylabel('Gap_ss/Gap_ee',fontsize=12)
		ax0.invert_xaxis()
		ax0.tick_params(labelsize = 12)
		ax0.set_yticks(np.arange(-7,8,1))
		plt.xlim([1000,0])
		plt.legend(frameon= False)

		# plt.title('Gap_ss/Gap_ee distribution')

		ax1 = plt.subplot(gs[1])
		tsplot(pv_lag_ss,lag_range,color= 'coral',line_color= 'orangered',label= 'Gap_ss [PV index]',plot_mean = True,plot_median=False)
		tsplot(pv_lag_ee,lag_range,color= 'dodgerblue',line_color= 'blue',label ='Gap_ee [PV index]',plot_mean = True,plot_median=False)
		ax1.set_yticklabels([])
		ax1.tick_params(labelsize = 12)
		ax1.set_yticks(np.arange(-7,8,1))
		plt.legend(frameon= False)
		# plt.title('Gap_ss/Gap_ee variabilty')
		plt.xlim([0,1500])
		fb_fig.suptitle('(b) Interquartile range of Gap_ss/Gap_ee',fontsize=12,y = 0.95)
		plt.show() 
		plt.tight_layout()
		# plt.show() 
		fb_fig.savefig(file_dir + '/paper_figure/summary of Gap_ss_Gap_ee_yearly_mean_25_75_new.pdf')

	else:
		fb_fig = plt.figure(figsize=(6, 5))
		plt.rcParams["font.family"] = "serif"
		# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
		gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
		ax0 = plt.subplot(gs[0])
		for year in range(39):
			plt.plot(np.squeeze(tm_lag_ss[:,year]),lag_range,color= 'coral',linewidth= 1,alpha = 0.3)
			plt.plot(np.squeeze(tm_lag_ee[:,year]),lag_range,color= 'dodgerblue',linewidth= 1,alpha = 0.3)
		plt.plot(np.squeeze(tm_ss_mean),lag_range,color= 'orangered',linewidth= 1.5,alpha = 0.85,label = 'Gap_ss [TM index]')
		plt.plot(np.squeeze(tm_ee_mean),lag_range,color= 'blue',linewidth= 1.5,alpha = 0.85,label = 'Gap_ee [TM index]')
		ax0.yaxis.tick_right()
		plt.ylabel('Gap_ss/Gap_ee',fontsize=12)
		ax0.invert_xaxis()
		ax0.tick_params(labelsize = 12)
		ax0.set_yticks(np.arange(-7,8,1))
		plt.xlim([1250,0])
		plt.legend(frameon= False)

		# plt.title('Gap_ss/Gap_ee variabilty')

		ax1 = plt.subplot(gs[1])
		for year in range(39):
			plt.plot(np.squeeze(pv_lag_ss[:,year]),lag_range,color= 'coral',linewidth= 1,alpha = 0.3)
			plt.plot(np.squeeze(pv_lag_ee[:,year]),lag_range,color= 'dodgerblue',linewidth= 1,alpha = 0.3)
		plt.plot(np.squeeze(pv_ss_mean),lag_range,color= 'orangered',linewidth= 1.5,alpha = 0.8,label = 'Gap_ss [PV index]')
		plt.plot(np.squeeze(pv_ee_mean),lag_range,color= 'blue',linewidth= 1.5,alpha = 0.8,label = 'Gap_ee [PV index]')
		ax1.set_yticklabels([])
		ax1.tick_params(labelsize = 12)
		ax1.set_yticks(np.arange(-7,8,1))
		plt.legend(frameon= False)
		# plt.title('Gap_ss/Gap_ee variabilty')
		plt.xlim([0,2000])
		fb_fig.suptitle('(a) Gap_ss/Gap_ee distributons for each year',fontsize=12,y = 0.95)
		# plt.tight_layout()
		plt.show() 
		plt.tight_layout()
		fb_fig.savefig(file_dir + '/paper_figure/summary of Gap_ss_Gap_ee_each_line_each_year_new.pdf')


def main_daily_gridnum():
	fsize  = 12
	south_east = False
	move_ave = False
	file_dir = '/home/user/Documents/research/project1'
	hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
	# hw_sum  = blo_sum_tm90

	''' initial submission'''
	# ta_90th = sio.loadmat(file_dir + '/temperature/hw_tamax_90_3d_35_75N_detrend_105_39_sig_only.mat')['hw']

	''' first revision '''
	ta_90th = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_detrend']
	ta_90th = np.transpose(ta_90th,[2,0,1])
	ta_90th = np.reshape(ta_90th,[92,39,80,180],order = 'F')


	ta_90th_ori = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_ori']
	ta_90th_ori = np.transpose(ta_90th_ori,[2,0,1])
	ta_90th_ori = np.reshape(ta_90th_ori,[92,39,80,180],order = 'F')
	hw_sum = ta_90th_ori

	# day = range(1,92)
	# year = range(1979,2018)


	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
	for day in range(hw_sum.shape[0]):
		for year in range(hw_sum.shape[1]):
			hw_sum[day,year,:,:][np.where(land_mask == 0 )] = 0
			ta_90th[day,year,:,:][np.where(land_mask == 0 )] = 0
			blo_sum_tm90 [day,year,:,:][np.where(land_mask == 0 )] = 0
			blo_sum_weak [day,year,:,:][np.where(land_mask == 0 )] = 0


	hw_tm = co_occur(ta_90th, blo_sum_tm90,False,False)
	hw_pv = co_occur(ta_90th, blo_sum_weak,False,False)

	# if south_east:
	# 	hw_sum = hw_sum[:,:,30:70,80:120]
	frequency = False
	concurrence =  True
	HWS_BLS = True
	tm = True
	pv = False
	d5 = True
	whole_region = True
	
	if frequency:
		hw_sum_list = [hw_sum,ta_90th,blo_sum_tm90,blo_sum_weak]
		title_list = ['(a) Heatwave grids (before detrending)', '(b) Heatwave grids (after detrending)', '(k) Blocking grids [TM index]', '(l) Blocking grids [PV index]']
		vmax_list  = [2500, 2500,2000,2000]
		xlim_list = [35000,35000,25000,25000]
		ylim_list = [120000,120000,60000,60000]


		if concurrence:
			# concurrence
			hw_sum_list = [hw_tm,hw_pv]
			title_list = ['(m) HW-TM grids [concurrence]', '(n) HW-PV grids [concurrence]']
			vmax_list  = [1000, 1000]
			xlim_list = [8000,8000]
			ylim_list = [25000,25000]

		# vmax_list  = [1, 1]
		# xlim_list = [20,20]
		# ylim_list = [40,40]
	
	elif HWS_BLS:
		for sub_reg in range(1):
			''' initial submisson'''
			# hw_pv_uni_2d = sio.loadmat(file_dir + '/result_for_figure/hw_pv_lag_arr_based_on_blo_15_'+ str(sub_reg)+'_ct.mat')['pv']
			# hw_tm_uni_2d = sio.loadmat(file_dir + '/result_for_figure/hw_pv_lag_arr_based_on_blo_15_'+ str(sub_reg)+'_ct.mat')['tm']

			''' first revision'''
			if not d5:
				# hw_pv_uni_2d = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_'+ str(sub_reg)+'_blo_new_ct.npy')
				# hw_tm_uni_2d = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_'+ str(sub_reg)+'_blo_new_ct.npy')

				hw_pv_uni_2d = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_'+ str(sub_reg)+'_hwd3_ct.npy')
				hw_tm_uni_2d = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_'+ str(sub_reg)+'_hwd3_ct.npy')

				# hw_pv_uni_2d = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_'+ str(sub_reg)+'_ct.npy')
				# hw_tm_uni_2d = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_'+ str(sub_reg)+'_ct.npy')

				if whole_region:
					hw_pv_uni_2d = np.load(file_dir + '/cpc_tmp/hw_blo_lag_range_based_on_blo_7_whole_region_ct.npz')['pv']
					hw_tm_uni_2d = np.load(file_dir + '/cpc_tmp/hw_blo_lag_range_based_on_blo_7_whole_region_ct.npz')['tm']

			else:
				hw_pv_uni_2d = np.load(file_dir + '/cpc_tmp/hw_pv_lag_arr_based_on_blo_7_'+ str(sub_reg)+'_hwd5_ct.npy')
				hw_tm_uni_2d = np.load(file_dir + '/cpc_tmp/hw_tm_lag_arr_based_on_blo_7_'+ str(sub_reg)+'_hwd5_ct.npy')
				
				if whole_region:
					sub_reg  = 0
					hw_pv_uni_2d = np.load(file_dir + '/cpc_tmp/hw_blo_lag_range_based_on_blo_7_whole_region_hwd5_ct.npz')['pv']
					hw_tm_uni_2d = np.load(file_dir + '/cpc_tmp/hw_blo_lag_range_based_on_blo_7_whole_region_hwd5_ct.npz')['tm']

			
			region_name = ['north','middle','south']
			
			if whole_region:
				# region_name = ['Entire study region']
				region_name = ['Whole']

			if tm:
				# hw_sum = hw_tm_uni_2d[8:23,8:23]
				hw_sum = hw_tm_uni_2d
				title_list = ['(a) HWS - TMS (North)','(c) HWS - TMS (Middle)','(e) HWS - TMS (South)']
				if not d5:
					vmax_list = [4000,280,800] # tm
					xlim_list = [15000,1000,3500] # tm
					ylim_list = [15000,1000,3500]

					# vmax_list = [5600,400,1200] # tm
					# xlim_list = [20000,1200,4000] # tm
					# ylim_list = [25000,2000,3500]
				
				else:
					vmax_list = [2000,240,280] # tm
					xlim_list = [6000,700,1200] # tm
					ylim_list = [6000,700,1200]

				if whole_region: 
					# title_list = ['Number of HWS-TMS pairs (Entire study region)']
					title_list = ['(g) HWS - TMS (Whole)']
					if not d5:
						vmax_list = [4400] # tm
						xlim_list = [20000] # tm
						ylim_list = [20000]
					else:
						vmax_list = [2500] # tm
						xlim_list = [8000] # tm
						ylim_list = [8000]

			if pv:
				# hw_sum = hw_pv_uni_2d[8:23,8:23]
				hw_sum = hw_pv_uni_2d
				title_list = ['(b) HWS - PVS (North)','(d) HWS - PVS (Middle)','(f) HWS - PVS (South)']
				if not d5:
					# vmax_list = [8000,1200,200] # pv
					# xlim_list = [30000,5000,600] # pv
					# ylim_list = [30000,5000,600]

					vmax_list = [7200,1200,160] # pv
					xlim_list = [28000,5000,400] # pv
					ylim_list = [25000,5000,500]
				else:
					vmax_list = [2600,450,100] # pv
					xlim_list = [10000,1800,300] # pv
					ylim_list = [10000,1800,300]

				if whole_region:
					# title_list = ['Number of HWS-PVS pairs (Entire study region)']
					title_list = ['(h) HWS - PVS (Whole)']
					if not d5:
						vmax_list = [7200] # pv
						xlim_list = [30000] # pv
						ylim_list = [30000]
					else:
						vmax_list = [2500] # pv
						xlim_list = [12000] # pv
						ylim_list = [12000]


		
	# for i in range(len(hw_sum_list)):
	for i in range(1):
		if frequency:
			hw_sum_daily_grid = daily_gridnum(hw_sum_list[i])
			# concurrence
			# hw_sum_daily_grid_tm = daily_gridnum(blo_sum_weak)
			# dif  = hw_sum_daily_grid-hw_sum_daily_grid_tm
			# dif[np.where(dif<= 0 )] = 0
			# hw_sum_daily_grid = np.divide(hw_sum_daily_grid,hw_sum_daily_grid_tm)
			# hw_sum_daily_grid[np.isposinf(hw_sum_daily_grid)] = np.nan
			# pdb.set_trace()
		else:
			hw_sum_daily_grid = hw_sum
		# hw_ta90 = daily_gridnum(ta_90th)
		# blo_tm = daily_gridnum(blo_sum_tm90)
		# blo_pv = daily_gridnum(blo_sum_weak)

		# hw_year = np.nansum(np.abs(hw_sum_daily_grid),axis = 0) # frequency 
		# hw_day = np.nansum(np.abs(hw_sum_daily_grid),axis = 1)

		hw_year = np.nansum(np.abs(hw_sum_daily_grid),axis = 0)
		hw_day = np.nansum(np.abs(hw_sum_daily_grid),axis = 1)

		# hw_year = np.nansum(hw_sum_daily_grid,axis = 0)
		# hw_day = np.nansum(hw_sum_daily_grid,axis = 1)
		# hw_year = hw_year[:,np.newaxis]
		# hw_day = hw_day[:,np.newaxis]
		# hw_year_df = pd.DataFrame(hw_year,columns = ['hw_year'])
		# hw_day_df = pd.DataFrame(hw_day,columns = ['hw_day'])
		if frequency:
			hw_year_df = pd.DataFrame(hw_year,columns = ['year'])
			hw_day_df = pd.DataFrame(hw_day,columns = ['day'])
		if HWS_BLS:
			# hw_year_df = pd.DataFrame(hw_year,columns = ['Gap_ss']) # wrong
			# hw_day_df = pd.DataFrame(hw_day,columns = ['Gap_ee'])

			hw_year_df = pd.DataFrame(hw_year,columns = ['Gap_ee'])
			hw_day_df = pd.DataFrame(hw_day,columns = ['Gap_ss'])

		# pdb.set_trace()
		if move_ave:
			for year in range(hw_sum.shape[1]):
				hw_year = hw_sum[:,year]
				hw_year = np.convolve(hw_year, np.ones((7,))/7, mode='same')
				hw_sum[:,year] = hw_year

		

		# Joint frequency distribion
		if frequency:
			# grid_fig = plt.figure(figsize=(6, 5))
			# gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3,1]) 
			
			grid_fig = plt.figure(figsize=(6, 4.8))
			gs = gridspec.GridSpec(2, 3, width_ratios=[3,0.1,1], height_ratios=[3.5,1]) 

			plt.rcParams["font.family"] = "serif"

			# ax0 = plt.subplot(gs[0])
			ax0 = plt.subplot(gs[0,:2])
			sns.set(style="white")
			rdylbu = brewer2mpl.get_map('RdYlBu', 'Diverging',11, reverse=True).mpl_colormap
			sns.heatmap(hw_sum_daily_grid, cmap=rdylbu, robust=False, annot=False, cbar= True, vmin=0,vmax=vmax_list[i],cbar_kws={"extend": "both"})
			ax0.set_xticks(np.arange(0,40,5))
			# ax0.set_xticklabels(np.arange(1979,2018,8));
			ax0.set_xticklabels(["79","84","89","94","99","04","09","14","18"]);
			plt.xticks(rotation=45)
			ax0.set_yticks(np.arange(0,92,15));
			ax0.set_yticklabels(np.arange(1,93,15));
			ax0.tick_params(labelsize = fsize)

			# plt.title('Frequency Distribution of daily gridnum',fontsize=12)
			plt.title(title_list[i],fontsize=fsize)
			plt.xlabel('Year',fontsize = fsize)
			plt.ylabel('Day',fontsize = fsize)
			plt.tight_layout()

			# Marginal frequency distribution of earnings
			plt.rcParams["font.family"] = "serif"
			# ax1 = plt.subplot(gs[1])
			ax1 = plt.subplot(gs[0,2])
			hw_day_df.plot(ax = plt.subplot(gs[0,2]),kind='barh',width=1, facecolor='gray', edgecolor='black').invert_yaxis()
			# hw_day_df.plot(ax = plt.subplot(gs[1]),color = 'grey').invert_yaxis()
			ax1.set_yticks(np.arange(0,92,15));
			ax1.set_yticklabels(np.arange(1,93,15));
			ax1.tick_params(labelsize = fsize)
			# plt.autoscale() # the gap between 
			# ax1.set_xlim([0,35000]) # heatwave
			ax1.set_xlim([0,xlim_list[i]]) # heatwave
			# pdb.set_trace()
			ax1.get_legend().remove()
			# ax1.set_xlim([0,25000]) # blocking 
			plt.ticklabel_format(axis='x',style = 'sci',scilimits= (0,0))
			plt.tight_layout()
			

			# Marginal frequency distribution of schooling
			# ax2 = plt.subplot(gs[2])
			ax2 = plt.subplot(gs[1,0])
			plt.rcParams["font.family"] = "serif"
			hw_year_df.plot(ax = plt.subplot(gs[1,0]),kind='bar',width=1, facecolor='gray', edgecolor='black')
			x = range(0,39)
			# pdb.set_trace()
			z = np.polyfit(x,hw_year,1)
			p = np.poly1d(z)
			plt.plot(x,p(x),color = 'k',linestyle = '-',linewidth= 1.5)
			ax2.set_xticks(np.arange(0,40,5))
			# ax2.set_xticklabels(np.arange(1979,2018,8));
			ax2.set_xticklabels(["79","84","89","94","99","04","09","14","18"]);
			plt.xticks(rotation = 45)
			ax2.tick_params(labelsize = fsize)
			ax2.get_legend().remove()
			# plt.autoscale()
			# ax2.set_ylim([0,120000]) # heatwave
			ax2.set_ylim([0,ylim_list[i]]) # heatwave
			# ax2.set_ylim([0,60000]) # blocking intense
			plt.ticklabel_format(axis='y',style = 'sci',scilimits= (0,0))
			plt.tight_layout()
			plt.show()
			''' initial submission ''' 
			# grid_fig.savefig(file_dir + '/paper_figure/figure1_4_hw_grids_new_' + str(i)+'.png')

			''' first revision'''
			grid_fig.savefig(file_dir + '/paper_figure/cpc_figure1_4_hw_grids_new2_co_occur' + str(i)+'.pdf')

		if HWS_BLS:
			grid_fig = plt.figure(figsize=(6, 4.8))
			gs = gridspec.GridSpec(2, 3, width_ratios=[3,0.1,1], height_ratios=[3,1]) 
			plt.rcParams["font.family"] = "serif"
			ax0 = plt.subplot(gs[0,:2])
			sns.set(style="white")

			clist =  [0,0.1,0.2,0.3,0.6,0.7,0.8,0.9]
			colors = plt.cm.RdYlBu_r(clist)
			cmap1 = ListedColormap(colors)

			vmax_tm = vmax_list[sub_reg]
			cticks_tm = np.linspace(0,vmax_tm,num = 9)
			sns.heatmap(hw_sum_daily_grid, cmap= cmap1, robust=False, annot=False, cbar= True, vmin=0,vmax = vmax_tm,cbar_kws={"extend": "neither","ticks":cticks_tm}) # unique event number

			ax0.set_xticks(np.arange(0.5,15.5,1))
			ax0.set_xticklabels(np.arange(-7,8,1));
			ax0.set_yticks(np.arange(0.5,15.5,1));
			ax0.set_yticklabels(np.arange(-7,8,1));
			ax0.tick_params(labelsize = fsize)

			# plt.title('Frequency Distribution of daily gridnum',fontsize=12)
			plt.title(title_list[sub_reg],fontsize=fsize)
			# plt.xlabel('HWS Start - BLS Start',fontsize = fsize)
			# plt.ylabel('HWS End - BLS End',fontsize = fsize)

			plt.xlabel('HWS End - BLS End',fontsize = fsize)
			plt.ylabel('HWS Start - BLS Start',fontsize = fsize)
			plt.tight_layout()

			# Marginal frequency distribution of earnings
			plt.rcParams["font.family"] = "serif"
			ax1 = plt.subplot(gs[0,2])
			hw_day_df.plot(ax = plt.subplot(gs[0,2]),kind='barh',width=1, facecolor='gray', edgecolor='black').invert_yaxis()
			# hw_day_df.plot(kind='barh',width=1, facecolor='gray', edgecolor='black').invert_yaxis()
			# hw_day_df.plot(ax = plt.subplot(gs[1]),color = 'grey').invert_yaxis()
			ax1.set_yticks(np.arange(0,15,1));
			ax1.set_yticklabels(np.arange(-7,8,1));
			ax1.tick_params(labelsize = fsize)
			# plt.autoscale()
			# ax1.set_xlim([0,35000]) # heatwave
			ax1.set_xlim([0,xlim_list[sub_reg]]) # heatwave
			# ax1.set_xlim([0,25000]) # blocking 
			plt.ticklabel_format(axis='x',style = 'sci',scilimits= (0,0))
			plt.tight_layout()
			

			# Marginal frequency distribution of schooling
			ax2 = plt.subplot(gs[1,0])
			plt.rcParams["font.family"] = "serif"
			hw_year_df.plot(ax = plt.subplot(gs[1,0]),kind='bar',width=1, facecolor='gray', edgecolor='black')
			ax2.set_xticks(np.arange(0,15,1));
			ax2.set_xticklabels(np.arange(-7,8,1));
			ax2.tick_params(labelsize = fsize)
			# plt.autoscale()
			# ax2.set_ylim([0,120000]) # heatwave
			ax2.set_ylim([0,ylim_list[sub_reg]]) # heatwave
			# ax2.set_ylim([0,60000]) # blocking intense
			plt.ticklabel_format(axis='y',style = 'sci',scilimits= (0,0))

			plt.tight_layout()
			plt.show()
			# grid_fig.savefig(file_dir + '/paper_figure/figure3_temporal_association_marginal_pv_new_' + str(i) + region_name[sub_reg]+'.png')

			grid_fig.savefig(file_dir + '/paper_figure/cpc_figure3_temporal_association_marginal_tm_whole_hwd5_' + str(i) + region_name[sub_reg]+'.pdf')
			# grid_fig.savefig(file_dir + '/paper_figure/cpc_figure3_temporal_association_marginal_tm_new_hwd5_new_' + str(i) + region_name[sub_reg]+'.pdf')
			# grid_fig.savefig(file_dir + '/paper_figure/cpc_figure3_temporal_association_marginal_pv_new_blo_hwd3_new_' + str(i) + region_name[sub_reg]+'.pdf')




def main_condi_pro():
	diff_ratio = False
	file_dir = '/home/user/Documents/research/project1'
	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']

	if diff_ratio:
		''' the difference between climotology and  P(blocking |heatwave)'''
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
		''' initial submission '''
		# path2 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P*_pv*_1.2_0.4_daily__tamax_90th_detrend__1000.mat')) # for manuscript 
		# path1 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P*_tm90_no_spatialfilter__tamax_90th_detrend__1000.mat'))

		# path2 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P*(blocking|heatwave)_pv*_1.2_0.4_daily__tamax_90th_detrend__1000.mat')) # for manuscript 
		# path1 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P*(blocking|heatwave)_tm90_no_spatialfilter__tamax_90th_detrend__1000.mat'))

		
		''' first revision'''
		# path2 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P*(heatwave|blocking)_pv*_1.2_0.40_daily__tamax_90th_detrend__1000.npy')) # for manuscript 
		# path1 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P*(heatwave|blocking)_tm90_no_spatialfilter__tamax_90th_detrend__1000.npy'))

		# path2 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P*(blocking|heatwave)_pv*_1.2_0.40_daily__tamax_90th_detrend__1000.npy')) # for manuscript 
		# path1 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P*(blocking|heatwave)_tm90_no_spatialfilter__tamax_90th_detrend__1000.npy'))


		''' sensitivity test -- first revision '''
		# path1 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P*(heatwave|blocking)_pv*_daily__tamax_90th_detrend__1000.npy'))  # pv
		# path2 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P*(blocking|heatwave)_pv*_daily__tamax_90th_detrend__1000.npy'))

		path2 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P(blocking|heatwave)_tm90_with_spatialfilter__tamax_90th_detrend__1000.npy')) # tm 
		path1 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P(heatwave|blocking)_tm90_with_spatialfilter__tamax_90th_detrend__1000.npy'))


		# path2 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P(blocking|heatwave)_tm90_no_spatialfilter__tamax_90th_detrend__1000.npy')) # tm 
		# path1 = sorted(glob.glob(file_dir + '/cpc_tmp/significant_mentocor_test_P(heatwave|blocking)_tm90_no_spatialfilter__tamax_90th_detrend__1000.npy'))


		# ''' initial submission '''
		# data1 = sio.loadmat(path1[0])['anom']
		# data2 = sio.loadmat(path2[0])['anom']

		# ''' initial submission  ''' 
		# hw_sum_3d, blo_sum_tm90_3d,blo_sum_weak_3d, hw_sum, blo_sum_tm90,blo_sum_weak = load_data()
		# blo_tm_freq = block_freq(blo_sum_tm90,dimen = 'grid')/92/39
		# blo_pv_freq = block_freq(blo_sum_weak,dimen = 'grid')/92/39
		# anom1 = np.divide(data1,blo_tm_freq)
		# anom2 = np.divide(data2,blo_pv_freq)

		''' first revision'''
		data1 = np.load(path1[0])
		data2 = np.load(path2[0])
		data1[np.where(land_mask==0)] = np.nan
		data2[np.where(land_mask==0)] = np.nan
		anom = False

		if anom:
			hw = np.load(file_dir + '/cpc_tmp/hw_cpc_3d.npz')['hw_detrend']
			hw = np.transpose(hw,[2,0,1])
			hw = np.reshape(hw,[92,39,80,180],order = 'F')

			hw_freq = block_freq(hw,dimen = 'grid')/92/39
			anom1 = np.divide(data1,hw_freq)
			anom2 = np.divide(data2,hw_freq)
			
			anom1[np.where(land_mask==0)] = np.nan
			anom2[np.where(land_mask==0)] = np.nan

			subreg = [[0,40],[40,60],[60,80]]
			data_mean  = np.zeros([3,2],dtype = np.float16)
			anom_mean = np.zeros([3,2],dtype = np.float16)

			for reg in range(3):
				# for the percentage of the concurrent association in the manuscirpt
				anom_s1 = anom1[subreg[reg][0]:subreg[reg][1],:]
				anom_s2 = anom2[subreg[reg][0]:subreg[reg][1],:]

				data_s1 = data1[subreg[reg][0]:subreg[reg][1],:]
				data_s2 = data2[subreg[reg][0]:subreg[reg][1],:]

				data_mean[reg,0] = np.nanmean(data_s1[~np.isnan(data_s1)])
				data_mean[reg,1] = np.nanmean(data_s2[~np.isnan(data_s2)])

				anom_mean[reg,0] = np.nanmean(anom_s1[~np.isnan(anom_s1)])
				anom_mean[reg,1] = np.nanmean(anom_s2[~np.isnan(anom_s2)])
			# pdb.set_trace()

			# pdb.set_trace()
			title1 = '(i) Relative anomaly [TM index]'
			title2 = '(j) Relative anomaly [PV index]'
			clevs = np.arange(1,11,1)
			single_data = False
			# clevs = np.arange(0,0.55,0.05)
			# outfig = file_dir + '/result_figure/conditional_hw_in_blo.png'
			# outfig = file_dir + '/result_figure/conditional_prob_' + pv_type + '_two_blocking_index.png'
			# outfig = file_dir + '/paper_figure/figure_2_3_' + str(i)+'.png'
			outfig = file_dir + '/paper_figure/cpc_Figure_2_anom.pdf'
			single_data = False
			geomap_china(anom1, anom2, clevs,clevs,title1,title2,outfig,single_data) # 4



		# path1 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P(heatwave|blocking)*pv*_*1000.mat')) # for sensitivity test
		# path2 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)*pv*_*1000.mat'))
		pv_list = [1.2,1.2,1.2,1.2,1.3,1.3,1.3,1.3]
		or_list = [0.35,0.40,0.45,0.50,0.35,0.40,0.45,0.50]
		# title_order = [['(a)','(b)'], ['(c)','(d)'],['(e)','(f)'],['(g)','(h)'],['(i)','(j)'],['(k)','(l)'], ['(m)','(n)'],['(o)','(p)'],['(q)','(r)'],['(s)','(t)']]
		# title_order = [['(o)','(p)'], ['(m)','(n)'],['(k)','(l)'],['(i)','(j)'],['(g)','(h)'],['(e)','(f)'], ['(c)','(d)'],['(a)','(b)']]
		title_order = [['(q)','(r)'],['(s)','(t)']]

		# path1 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P(heatwave|blocking)_*tm90*1000.mat'))
		# path2 = sorted(glob.glob(file_dir + '/temporal_result/significant_mentocor_test_P(blocking|heatwave)_*tm90*1000.mat'))


		for i in range(len(path1)):
			print(path1[i])
			# pv_type = path1[i][102:120]
			'''for manuscript'''
			# if i ==0:
			# 	title1 = '(g) ' + path1[i][81:101]
			# 	title2 = '(h) ' + path2[i][81:101]
			# 	# title1 = '(g) P(blocking|heatwave)
			# 	# title2 = '(h) ' + path2[i][81:101]
			# else:
			# 	title1 = '(e) ' + path1[i][81:101]
			# 	title2 = '(f) ' + path2[i][81:101]

			''' for sensitivity test '''
			# title1 = title_order[i][0] +' '+ path1[i][81:101]
			# title2 = title_order[i][1]+' ' + path2[i][81:101]

			''' first revision '''
			# title1 = '(e) P(heatwave|blocking) [TM index]'
			# title2 = '(f) P(heatwave|blocking) [PV index]'

			# title1 = '(g) P(blocking|heatwave) [TM index]'
			# title2 = '(h) P(blocking|heatwave) [PV index]'

			''' for PV '''
			# title1 = title_order[i][0] + ' P(heatwave|blocking) [' + 'PV < -' + str(pv_list[i]) + ' pvu, OR =' + str(or_list[i]) + ']' 
			# title2 = title_order[i][1] + ' P(blocking|heatwave) [' + 'PV < -' + str(pv_list[i]) + ' pvu, OR =' + str(or_list[i]) + ']'

			''' for TM'''
			title1 = '(s) P(heatwave|blocking) [TM, with spatial filter]' 
			title2 = '(t) P(blocking|heatwave) [TM, with spatial filter]'

			# title1 = '(q) P(heatwave|blocking) [TM, no spatial filter]' 
			# title2 = '(r) P(blocking|heatwave) [TM, no spatial filter]'


			# pdb.set_trace()

			# # ''' initial submission '''
			# data1 = sio.loadmat(path1[i])['anom']
			# data2 = sio.loadmat(path2[i])['anom']

			''' first revision'''
			data1 = np.load(path1[i])
			data2 = np.load(path2[i])

			data1[np.where(land_mask==0)] = np.nan
			data2[np.where(land_mask==0)] = np.nan
			clevs = np.arange(0,0.55,0.05)
			# clevs = np.arange(0,0.55,0.05)
			# outfig = file_dir + '/result_figure/conditional_hw_in_blo.png'
			# outfig = file_dir + '/result_figure/conditional_prob_' + pv_type + '_two_blocking_index.png'
			# outfig = file_dir + '/paper_figure/figure_2_3_' + str(i)+'.png'
			'''first revision'''
			# outfig = file_dir + '/paper_figure/cpc_Figure_blo_in_hw_tm_sensi_new_reverse_' + str(i)+'.pdf'
			outfig = file_dir + '/paper_figure/cpc_Figure_blo_in_hw_tm_sensi_with_filter_tight.pdf'
			# outfig = file_dir + '/paper_figure/cpc_Figure_hw_in_blo_manu.pdf'
			single_data = False
			pdb.set_trace()
			geomap_china(data1, data2, clevs,clevs,title1,title2,outfig,single_data) # 4



def temporal_association():
	''' temporal association '''
	file_dir = '/home/user/Documents/research/project1'
	fsize = 13
	region_name = ['north','middle','south']
	multi_colorbar = False
	range0 = 15
	title_list = [['(a) HWS - TMS', '(b) HWS - PVS'],['(c) HWS - TMS','(d) HWS - PVS'],['(e) HWS - TMS','(f) HWS - PVS']]
	for sub_reg in range(2,3):

		hw_pv_uni_2d = sio.loadmat(file_dir + '/result_for_figure/hw_pv_lag_arr_based_on_blo_15_'+ str(sub_reg)+'_ct.mat')['pv']
		hw_tm_uni_2d = sio.loadmat(file_dir + '/result_for_figure/hw_pv_lag_arr_based_on_blo_15_'+ str(sub_reg)+'_ct.mat')['tm']

		tm_bars = [4,0.28,0.8]
		pv_bars = [8,1.2,0.2]
		
		num_fig = plt.figure(figsize=(12, 5))

		outfig = file_dir + '/paper_figure/ hw_blo_heatmap_ss_ee_' + region_name[sub_reg]+'_unique_hw_blo_pairs_new.png'
		gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
		plt.rcParams["font.family"] = "serif"
		ax0 = plt.subplot(gs[0])
		hw_tm_uni_2d = hw_tm_uni_2d [8:23,8:23]

		clist =  [0,0.1,0.2,0.3,0.6,0.7,0.8,0.9]
		colors = plt.cm.RdYlBu_r(clist)
		cmap1 = ListedColormap(colors)

		vmax_tm = tm_bars[sub_reg]*1000
		cticks_tm = np.linspace(0,vmax_tm,num = 9)
		sns.heatmap(hw_tm_uni_2d, cmap= cmap1, robust=False, annot=False, cbar= True, vmin=0,vmax = vmax_tm,cbar_kws={"extend": "neither","ticks":cticks_tm}) # unique event number

		plt.title(title_list[sub_reg][0],fontsize=fsize)
		plt.xlabel('HWS-Start - BLS-Start',fontsize=fsize)
		plt.ylabel('HWS-End - BLS-End',fontsize=fsize)
		ax0.tick_params(labelsize = fsize)

		if multi_colorbar:
			plt.rcParams["font.family"] = "serif"
			ax01= ax0.twinx()
			ax01.set_ylim([0,5])
			newlabel01 = count_tm[sub_reg]
			ct2pert = lambda t: t*1e5/tm_reg_num[sub_reg] # convert function: from Kelvin to Degree Celsius"ticks":cticks}
			newpos01 = [ct2pert(t) for t in newlabel01]   # position of the ticklabels in the old axis
			ax01.set_yticks(newpos01)			
			print(newpos01)
			ax01.tick_params(axis='y')
			ax01.spines['right'].set_color('steelblue')
			ax01.set_yticklabels(newlabel01)
			ax01.yaxis.set_ticks_position('right') # set the position of the second axis to right
			ax01.yaxis.set_label_position('right') 
			ax01.spines['right'].set_position(('outward', 110))
			# ax01.set_ylabel('Count-TM (1e3)')

			plt.rcParams["font.family"] = "serif"
			ax02= ax0.twinx()
			ax02.set_ylim([0,5])
			newlabel02 = hw_per_tm[sub_reg]
			ct2pert = lambda t: t*hw_reg_num[sub_reg]/tm_reg_num[sub_reg] # convert function: from Kelvin to Degree Celsius
			newpos02   = [ct2pert(t) for t in newlabel02]   # position of the ticklabels in the old axis
			ax02.set_yticks(newpos02)
			print(newpos02)
			ax02.tick_params(axis='y', color = 'coral',labelcolor='coral')
			ax02.spines['right'].set_color('coral')
			ax02.set_yticklabels(newlabel02)
			ax02.yaxis.set_ticks_position('right') # set the position of the second axis to right
			ax02.yaxis.set_label_position('right') 
			ax02.spines['right'].set_position(('outward', 65))

		if range0 == -30:
			ax0.set_xticks(np.arange(0,10,2))
			ax0.set_xticklabels(np.arange(-10,10,4));
			ax0.set_yticks(np.arange(0,10,2));
			ax0.set_yticklabels(np.arange(-10,10,4));
			ax0.tick_params(labelsize = 12)
		else:
			ax0.set_xticks(np.arange(0.5,15.5,1))
			ax0.set_xticklabels(np.arange(-7,8,1));
			ax0.set_yticks(np.arange(0.5,15.5,1));
			ax0.set_yticklabels(np.arange(-7,8,1));
			ax0.tick_params(labelsize = fsize)
			
			# ax0.set_xticks(np.arange(0,15,1))
			# ax0.set_xticklabels(np.arange(-7,8,1));
			# ax0.set_yticks(np.arange(0,15,1));
			# ax0.set_yticklabels(np.arange(-7,8,1));
			# ax0.tick_params(labelsize = fsize)

		plt.tight_layout()

		plt.rcParams["font.family"] = "serif"
		ax1 = plt.subplot(gs[1])
		ax1.tick_params(labelsize = fsize)
		sns.set(style="white")
		hw_pv_uni_2d = hw_pv_uni_2d [8:23,8:23]
		vmax_pv = pv_bars[sub_reg]*1000
		cticks_pv = np.linspace(0,vmax_pv,num = 9)
		sns.heatmap(hw_pv_uni_2d, cmap= cmap1, robust=False, annot=False, cbar= True, vmin=0,vmax=vmax_pv,cbar_kws={"extend": "neither","ticks":cticks_pv})

		plt.title(title_list[sub_reg][1],fontsize=fsize)
		plt.xlabel('HWS-Start - BLS-Start',fontsize=fsize)
		plt.ylabel('HWS-End - BLS-End',fontsize=fsize)

		# plt.xlabel('HW-Start - BL-End',fontsize=12)
		# plt.ylabel('HW-End - BL-Start',fontsize=12)

		if multi_colorbar:
			# colorbar 1
			plt.rcParams["font.family"] = "serif"
			ax11= ax1.twinx()
			# ax11.set_ylim([0,count_pv[sub_reg][-1]])
			ax11.set_ylim([0,5])
			newlabel11 = count_pv[sub_reg]
			ct2pert = lambda t: t*1e5/pv_reg_num[sub_reg] # convert function: from Kelvin to Degree Celsius
			newpos11   = [ct2pert(t) for t in newlabel11]
			print(newpos11)   # position of the ticklabels in the old axis
			ax11.tick_params(axis='y', color = 'steelblue',labelcolor='steelblue')
			ax11.spines['right'].set_color('steelblue')
			ax11.set_yticks(newpos11)
			ax11.set_yticklabels(newlabel11)

			ax11.yaxis.set_ticks_position('right') # set the position of the second axis to right
			ax11.yaxis.set_label_position('right') 
			ax11.spines['right'].set_position(('outward', 110))
			# ax11.set_ylabel('Count-PV (1e3)')

			# colorbar 2
			plt.rcParams["font.family"] = "serif"
			ax12= ax1.twinx()
			# ax12.set_ylim([0,hw_per_pv[sub_reg][-1]])
			ax12.set_ylim([0,5])
			newlabel12 = hw_per_pv[sub_reg]
			ct2pert = lambda t: t*hw_reg_num[sub_reg]/pv_reg_num[sub_reg] # convert function: from Kelvin to Degree Celsius
			newpos12   = [ct2pert(t) for t in newlabel12] 
			print(newpos12)  # position of the ticklabels in the old axis
			ax12.tick_params(axis='y', color = 'coral',labelcolor='coral')
			ax12.spines['right'].set_color('coral')
			ax12.set_yticks(newpos12)
			ax12.set_yticklabels(newlabel12)
			ax12.yaxis.set_ticks_position('right') # set the position of the second axis to right
			ax12.yaxis.set_label_position('right') 
			ax12.spines['right'].set_position(('outward', 65))

		if range0 == -30:
			ax1.set_xticks(np.arange(0,10,2))
			ax1.set_xticklabels(np.arange(-10,10,4));
			ax1.set_yticks(np.arange(0,10,2));
			ax1.set_yticklabels(np.arange(-10,10,4));
			ax1.tick_params(labelsize = 12)
		else:
			# ax1.set_xticks(np.arange(0,15,1))
			# ax1.set_xticklabels(np.arange(-7,8,1));
			# ax1.set_yticks(np.arange(0,15,1));
			# ax1.set_yticklabels(np.arange(-7,8,1));
			# # ax1.tick_params(labelsize = 12)
			# ax1.tick_params(labelsize = fsize)

			ax1.set_xticks(np.arange(0.5,15.5,1))
			ax1.set_xticklabels(np.arange(-7,8,1));
			ax1.set_yticks(np.arange(0.5,15.5,1));
			ax1.set_yticklabels(np.arange(-7,8,1));
			ax1.tick_params(labelsize = fsize)
			# ax1.set_xticks(np.arange(0,16,2))
			# ax1.set_xticklabels(np.arange(-12,3,2));
			# ax1.set_yticks(np.arange(0,16,2));
			# ax1.set_yticklabels(np.arange(-3,12,2));
			# ax1.tick_params(labelsize = 12)

		plt.tight_layout()
		print('total affec area')
		plt.show()
		num_fig.savefig(outfig)



def sens_blo_freq():
	file_dir = '/home/user/Documents/research/project1'
	blo_sum_weak1 = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.3_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.5_daily_extent_100.mat')['blocking']
	blo_sum_weak2 = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.3_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.45_daily_extent_100.mat')['blocking']
	blo_sum_weak3 = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.3_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.4_daily_extent_100.mat')['blocking']
	blo_sum_weak4 = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.3_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.35_daily_extent_100.mat')['blocking']
	
	blo_sum_weak5 = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.5_daily_extent_100.mat')['blocking']
	blo_sum_weak6 = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.45_daily_extent_100.mat')['blocking']
	blo_sum_weak7 = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.4_daily_extent_100.mat')['blocking']
	blo_sum_weak8 = sio.loadmat(file_dir +  '/blocking_event/summer_north_asia_daily_blocking_pv1.2_5day_2dtrack_weighted_4d_35_75N_all_ratio_0.35_daily_extent_100.mat')['blocking']

	blo_sum_tm902 = sio.loadmat(file_dir + '/blocking/summer_blocking_spatial_filter_5day_1979_2017_0_90N_valid.mat')['blocking'] # TM 90 0_90 N north 
	blo_sum_tm901 = sio.loadmat(file_dir + '/blocking/summer_blocking_nospatial_filter_5day_1979_2017_0_90N_valid.mat')['blocking']  # no spatial filter
	
	blo_sum_tm901 = blo_sum_tm901[30:110,20:-1,:] # 75-20N
	blo_sum_tm901 = np.transpose(blo_sum_tm901,[2,0,1])
	blo_sum_tm901 = np.reshape(blo_sum_tm901,[92,39,blo_sum_tm901.shape[1],blo_sum_tm901.shape[2]],order='F') # 4-dimention

	blo_sum_tm902 = blo_sum_tm902[30:110,20:-1,:] # 75-20N
	blo_sum_tm902 = np.transpose(blo_sum_tm902,[2,0,1])
	blo_sum_tm902 = np.reshape(blo_sum_tm902,[92,39,blo_sum_tm902.shape[1],blo_sum_tm902.shape[2]],order='F') # 4-dimention


	b1 = block_freq(blo_sum_weak1,dimen = 'grid')/39/92
	b2 = block_freq(blo_sum_weak2,dimen = 'grid')/39/92
	b3 = block_freq(blo_sum_weak3,dimen = 'grid')/39/92
	b4 = block_freq(blo_sum_weak4,dimen = 'grid')/39/92

	b5 = block_freq(blo_sum_weak5,dimen = 'grid')/39/92
	b6 = block_freq(blo_sum_weak6,dimen = 'grid')/39/92
	b7 = block_freq(blo_sum_weak7,dimen = 'grid')/39/92
	b8 = block_freq(blo_sum_weak8,dimen = 'grid')/39/92


	b9 = block_freq(blo_sum_tm901,dimen = 'grid')/39/92
	b10 = block_freq(blo_sum_tm902,dimen = 'grid')/39/92



	
	land_mask = sio.loadmat(file_dir + '/landmask/land_mask_35_75N_70_160E.mat')['mask']
	b1[np.where(land_mask ==0)] = np.nan
	b2[np.where(land_mask ==0)] = np.nan
	b3[np.where(land_mask ==0)] = np.nan
	b4[np.where(land_mask ==0)] = np.nan
	b5[np.where(land_mask ==0)] = np.nan
	b6[np.where(land_mask ==0)] = np.nan
	b7[np.where(land_mask ==0)] = np.nan
	b8[np.where(land_mask ==0)] = np.nan
	b9[np.where(land_mask ==0)] = np.nan
	b10[np.where(land_mask ==0)] = np.nan


	clevs = np.arange(0,0.11,0.01)
	title_order = [['(a)','(b)'], ['(c)','(d)'],['(e)','(f)'],['(g)','(h)'],['(i)','(j)']]

	bln_list = [['[PV < -1.3 pvu, OR = 0.50]','[PV < -1.3 pvu, OR = 0.45]'],['[PV < -1.3 pvu, OR = 0.40]','[PV < -1.3 pvu, OR = 0.35]'],['[PV < -1.2 pvu, OR = 0.50]','[PV < -1.2 pvu, OR = 0.45]'],['[PV < -1.2 pvu, OR = 0.40]','[PV < -1.3 pvu, OR = 0.35]'],['[TM, no spatial filter]','[TM, with spatial filter]']]
	blf_list = [[b1,b2],[b3,b4],[b5,b6],[b7,b8],[b9,b10]]

	for i in range(4,5):
		outfig = file_dir + '/paper_figure/figure_s1_blf_'+str(i) +'.pdf'
		single_data = False
		title1  = title_order[i][0] + ' Blocking frequency' + ' '+ bln_list[i][0]
		title2  = title_order[i][1]+' Blocking frequency' + ' ' + bln_list[i][1]

		geomap_china(blf_list[i][0], blf_list[i][1], clevs,clevs,title1,title2,outfig,single_data) 


# hwf_hws()
# hw_trend_line()
# main_daily_gridnum()
# main_condi_pro()
# temporal_association()
# sens_blo_freq()

hw_bl_var_line()
# tsplot_lag()