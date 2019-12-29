
import prettyplotlib as ppl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pdb
import numpy
import conda
import os
import seaborn as sns
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap, cm # conda install
# from mpltoolkits.basemap import Basemap, cm # conda install
import numpy as np
import matplotlib.pyplot as plt
from prettyplotlib import brewer2mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#from mk_change_point import Kendall_change_point_detection,Pettitt_change_point_detection,Buishand_U_change_point_detection
#from ncread_pv1_co import pv_overlap_check,pv_thres  # for the big event detection



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
    ax.voxels(data, edgecolors='darkgray',facecolors = 'steelblue')

    ax.tick_params(labelsize = 12)
    plt.tight_layout()
    fig.savefig(outfig)

    
def tsplot(x, y, n=20, percentile_min=25, percentile_max=75, color='r', plot_mean=False, plot_median=True, line_color='k',line_style ='-',label = True,lat = True,**kwargs):
    # calculate the lower and upper percentile groups, skipping 50 percentile
	if lat:
		# pdb.set_trace()
		nan_idx = np.squeeze(np.sum(np.isnan(x),axis = 1)/180)
		x[np.where(nan_idx>0.75)[0],:]= np.nan
		# pdb.set_trace()
		perc1 = np.nanpercentile(x, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=1)
		perc2 = np.nanpercentile(x, np.linspace(50, percentile_max, num=n+1)[1:], axis=1)
	    # pdb.set_trace()

		if 'alpha' in kwargs:
			alpha = kwargs.pop('alpha')
		else:
			# alpha = 1/n
			alpha = 0.08
	    # fill lower and upper percentile groups
		for p1, p2 in zip(perc1, perc2):
			plt.fill_betweenx(y,p1, p2,alpha=alpha, color=color, edgecolor=None) # fill_between two x


		if plot_mean:
			plt.plot(np.nanmean(x, axis=1),y, color=line_color, linestyle= line_style,linewidth = 2,label = label)


		if plot_median:
			plt.plot(np.nanmedian(x, axis=1),y, color=line_color,linestyle= line_style,linewidth = 2,label = label)
	    
			return plt.gca()

	else:
		perc1 = np.nanpercentile(y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
		perc2 = np.nanpercentile(y, np.linspace(50, percentile_max, num=n+1)[1:], axis=0)
	    # pdb.set_trace()

		if 'alpha' in kwargs:
			alpha = kwargs.pop('alpha')
		else:
			alpha = 1/n
	    # fill lower and upper percentile groups
		for p1, p2 in zip(perc1, perc2):
			plt.fill_between(x,p1, p2,alpha=alpha, color=color, edgecolor=None) # fill_between two x


		if plot_mean:
			plt.plot(x,np.nanmean(y, axis=0),color=line_color, linestyle= line_style,linewidth = 2,label = label)


		if plot_median:
			plt.plot(x,np.nanmedian(y, axis=0),color=line_color,linestyle= line_style,linewidth = 2,label = label)
	    
		return plt.gca()

def tsplot_back(x, y, n=20, percentile_min=1, percentile_max=99, color='r', plot_mean=True, plot_median=False, line_color='k', **kwargs):
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.percentile(y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
    perc2 = np.percentile(y, np.linspace(50, percentile_max, num=n+1)[1:], axis=0)

    if 'alpha' in kwargs:
        alpha = kwargs.pop('alpha')
    else:
        alpha = 1/n
    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        plt.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)


    if plot_mean:
        plt.plot(x, np.mean(y, axis=0), color=line_color)


    if plot_median:
        plt.plot(x, np.median(y, axis=0), color=line_color)
    
    return plt.gca()
    

def plot_cdf(data1,data2,num_bins1,num_bins2, title=None, xlabel=None,outfig = None):
 
    counts1, bin_edges1 = np.histogram (data1, bins=num_bins1, normed=True)
    counts2, bin_edges2 = np.histogram (data2, bins=num_bins2, normed=True)
    cdf1 = np.cumsum(counts1)
    cdf2 = np.cumsum(counts2)

    # sns.set_style("darkgrid")
    fig = plt.figure(figsize=(12,8))
    plt.plot (bin_edges1[1:], cdf1/cdf1[-1])
    plt.plot (bin_edges2[1:], cdf2/cdf2[-1])
    plt.ylabel('ECDF')
    plt.tight_layout()
   
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if outfig:
    	fig.savefig(outfig)
    # plt.show()


def plot_grid(data,vmax_v,vmin_v,out_title,outfig):
	fig1, ax= plt.subplots(1)
	ax.set_xticks(np.arange(0,40,4))
	ax.set_xticklabels(np.arange(1979,2018,4));
	ax.set_yticks(np.arange(0,92,9));
	ax.set_yticklabels(np.arange(1,93,9));

	# ax.set_xticks(np.arange(0,15400,2200))
	# ax.set_xticklabels(np.arange(0,15400,2200))
	# ax.set_yticks(np.arange(0,15400,2200))
	# ax.set_yticklabels(np.arange(0,15400,2200))

	# ax.set_yticks(np.arange(0,15,3));
	# ax.set_yticklabels(np.arange(3,18,3));

	# ax.set_xticks(np.arange(0,140,10))
	# ax.set_xticklabels(np.arange(70,140,5));
	# # ax.set_yticks(np.arange(1,93,9));

	# ax.set_yticks(np.arange(0,110,10));
	# ax.set_yticklabels(np.arange(10,55,5));

	rdylbu = brewer2mpl.get_map('RdYlBu', 'Diverging',11, reverse=True).mpl_colormap

	# rdylbu = brewer2mpl.get_map('Spectral', 'Diverging',11, reverse=True).mpl_colormap
	ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
	ppl.pcolormesh(fig1,ax,data,cmap = rdylbu,vmax = vmax_v,vmin = vmin_v)
	plt.title(out_title)
	plt.show()
	fig1.savefig(outfig)


def geo_grid_sig(data1,data2,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs,title1,title2,outfig,sig_level):
	fsize = 13
	clist =  [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
	colors = plt.cm.RdYlBu_r(clist)
	cmap1 = ListedColormap(colors)
	# fig = plt.figure(figsize=(12,8))
	fig = plt.figure(figsize=(6,5))
	plt.rcParams["font.family"] = "serif"

	m = Basemap(width=8000000,height=5000000,
            resolution='l',projection='eqdc',\
            lat_1=50.,lat_2=60,lat_0=lat_0,lon_0=lon_0)
	m.drawcoastlines()
	m.drawstates()
	m.drawcountries()
	parallels = np.arange(35.,80,10.)
	m.drawparallels(parallels,labels=[1,0,0,0],fontsize=fsize)
	# draw meridians
	meridians = np.arange(70.,180.,30.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fsize)

	ny = data1.shape[0]; nx = data1.shape[1]
	lon_1d = np.array(np.arange(70.25,160,0.5))
	lons = np.repeat(lon_1d[np.newaxis,:],80,axis=0)
	lat_1d = np.array(np.arange(35.25,75,0.5))
	lats = np.repeat(lat_1d[:,np.newaxis],180,axis=1)

	x, y = m(lons, lats) # compute map proj coordinates. 
	cs = m.contourf(x,y,data1,clevs,cmap=plt.cm.RdYlBu_r,extend ='both') ## 
	# cs = m.contourf(x,y,data1,clevs,cmap=cmap1,extend ='both') ## 
	m.plot(x[np.where(data2<sig_level)],y[np.where(data2<sig_level)],marker = '+', color = 'orangered',markersize = 0.6,fillstyle = 'none',linewidth=0)
	plt.title(title1,fontsize= fsize)
	# *********** ori ***************
	# cs = m.contourf(x,y,data,clevs,cmap=plt.cm.Set3)
	# add colorbar.
	cbar = m.colorbar(cs,location='bottom',pad="10%",extend='both')

	# cbar_ax =  fig.add_axes([0.08,0.1,0.84,0.035])
	# cbar = plt.colorbar(cs,cax=cbar_ax,orientation ='horizontal',extend='both')
	cbar.ax.tick_params(labelsize = fsize)
	plt.tight_layout(pad=3, w_pad=3)
	plt.show()
	fig.savefig(outfig)


def geo_grid_new2(data1,data2,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs,title1,title2,outfig):
	fsize = 13 
	clist =  [0,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9]
	colors = plt.cm.RdYlBu_r(clist)
	cmap1 = ListedColormap(colors)

	# rc('font',**{'family':'serif','serif':['Palatino']})
	# rc('text',usetex = False)
	fig = plt.figure(figsize=(12, 5))
	# fig = plt.figure()
	# plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams["font.family"] = "serif"

	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
	ax0 = plt.subplot(gs[0])
	m = Basemap(width=8000000,height=5000000,
        resolution='l',projection='eqdc',\
        lat_1=50.,lat_2=60,lat_0=lat_0,lon_0=lon_0)
	m.drawcoastlines()
	m.drawstates()
	m.drawcountries()
	#*********  draw parallels ************
	parallels = np.arange(35.,80,10.)
	# parallels = np.arange(35.,80,5.)
	m.drawparallels(parallels,labels=[1,0,0,0],fontsize=fsize,linewidth = 1)
	# ********** draw meridians ***********
	meridians = np.arange(70.,180.,30.)
	# meridians = np.arange(70.,180.,15.)
	# meridians = np.arange(70.,160.,10.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fsize,linewidth = 1)

	ny = data1.shape[0]; nx = data1.shape[1]
	lon_1d = np.array(np.arange(70.25,160,0.5))
	lons = np.repeat(lon_1d[np.newaxis,:],80,axis=0)
	lat_1d = np.array(np.arange(35.25,75,0.5))
	# lat_1d = np.array(np.arange(74.75,35,-0.5)) # why?
	lats = np.repeat(lat_1d[:,np.newaxis],180,axis=1)
	x, y = m(lons, lats) # compute map proj coordinates.
	# pdb.set_trace() 
	cs = m.contourf(x,y,data1,clevs,cmap=plt.cm.RdYlBu_r,extend = 'both') 
	# cs = m.contourf(x,y,data1,clevs,cmap=plt.get_cmap("RdYlBu_r",10),extend = 'both') 
	plt.title(title1,fontsize= fsize)
	# cbar_ax =  fig.add_axes([0.08,0.1,0.84,0.035])
	plt.tight_layout(pad=3, w_pad=3)
	
	ax1 = plt.subplot(gs[1])
	m = Basemap(width=8000000,height=5000000,
        resolution='l',projection='eqdc',\
        lat_1=50.,lat_2=60,lat_0=lat_0,lon_0=lon_0)
	m.drawcoastlines()
	m.drawstates()
	m.drawcountries()
	#*********  draw parallels ************
	parallels = np.arange(35.,80,10.)
	# parallels = np.arange(35.,80,5.)
	m.drawparallels(parallels,labels=[0,1,0,0],fontsize=fsize,linewidth = 1)
	# ********** draw meridians ***********
	meridians = np.arange(70.,180.,30.)
	# meridians = np.arange(70.,180.,15.)
	# meridians = np.arange(70.,160.,10.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fsize,linewidth = 1)

	ny = data2.shape[0]; nx = data2.shape[1]
	lon_1d = np.array(np.arange(70.25,160,0.5))
	lons = np.repeat(lon_1d[np.newaxis,:],80,axis=0)
	lat_1d = np.array(np.arange(35.25,75,0.5))
	# lat_1d = np.array(np.arange(74.75,35,-0.5))
	lats = np.repeat(lat_1d[:,np.newaxis],180,axis=1)
	x, y = m(lons, lats) # compute map proj coordinates. 
	cs = m.contourf(x,y,data2,clevs,cmap=plt.cm.RdYlBu_r,extend = 'both') 
	# cs = m.contourf(x,y,data2,clevs,cmap=plt.get_cmap("RdYlBu_r",10),extend = 'both') 
	plt.title(title2,fontsize= fsize)
	# cbar_ax =  fig.add_axes([0.12,0.08,0.78,0.03])
	# cbar_ax =  fig.add_axes([0.12,0.1,0.78,0.035])
	cbar_ax =  fig.add_axes([0.08,0.1,0.84,0.035])
	# cbar = m.colorbar(cs,location=cbar_ax)
	# fig.colorbar(cs,cax=cbar_ax)
	cbar = plt.colorbar(cs,cax=cbar_ax,orientation ='horizontal',extend='both')

	cbar.ax.tick_params(labelsize = fsize)
	plt.tight_layout(pad=3, w_pad=3)
	# plt.tight_layout()
	plt.show() 
	fig.savefig(outfig)


def geo_grid_2(data1,data2,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs1,clevs2,out_title,outfig):

	fig = plt.figure(figsize=(12,8))
	plt.rcParams["font.family"] = "serif"
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	m = Basemap(projection='cyl',\
	            llcrnrlat=lllat,urcrnrlat=urlat,\
	            llcrnrlon=lllon,urcrnrlon=urlon,resolution='l')

	m.drawcoastlines(color = 'grey')
	m.drawstates(color = 'grey')
	m.drawcountries(color = 'grey')

	parallels = np.arange(0.,90,10.)
	m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,color = 'grey')
	meridians = np.arange(0.,180.,10.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,color = 'grey')


	ny = data1.shape[0]; nx = data1.shape[1]
	lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
	x, y = m(lons, lats) # compute map proj coordinates.
	# clevs1 = np.arange(np.int(data2.min()),np.int(data2.max()),200)
	# clevs1 = np.arange(-100,2300,200)
	# pdb.set_trace()
	
	# ******************* back *******************
	cs1 = m.contour(x,y,data1,clevs1,colors='b',linestyles='solid',linewidths = 1.5) # contour line 
	plt.clabel(cs1, inline=1, fontsize= 10, fmt = '%d')
	cs2 = m.contour(x,y,data2,clevs2,colors='r',linestyles='dashed',linewidths = 1.5) # contour line # colors= 'y'
	plt.clabel(cs2, inline=1, fontsize= 10, fmt = '%d')

	# cs1 = m.contourf(x,y,data1,clevs1,colors='b',linestyles='solid',linewidths = 1.5) # contour line 
	# plt.clabel(cs1, inline=1, fontsize= 10, fmt = '%d')
	# cs2 = m.contourf(x,y,data2,clevs2,colors='r',linestyles='dashed',linewidths = 1.5) # contour line # colors= 'y'
	# plt.clabel(cs2, inline=1, fontsize= 10, fmt = '%d')
	# cbar = m.colorbar(cs3,location='bottom',pad="5%")
	plt.title(out_title)
	plt.tight_layout()
	plt.show()
	fig.savefig(outfig)


def geo_grid(data,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs,out_title,outfig):
	fsize  = 13
	# lon_0 = 105 #  
	# latcorners = nc.variables['lat'][:]
	# loncorners = -nc.variables['lon'][:]

	# lon_0 = -nc.variables['true_lon'].getValue()
	# lat_0 = nc.variables['true_lat'].getValue()
	# create figure and axes instances
	# fig = plt.figure(figsize=(12,8))
	fig = plt.figure(figsize=(6,5))
	plt.rcParams["font.family"] = "serif"

	# ax = fig.add_axes([0.1,0.1,0.8,0.8])

	# create polar stereographic Basemap instance.
	# llcrnrlon longitude of lower left hand corner of the desired map domain 
	# m = Basemap(projection='geos',lon_0=lon_0,lat_0=lat_0,\
	#             llcrnrlat=lllat,urcrnrlat=urlat,\
	#             llcrnrlon=lllon,urcrnrlon=120,\
	#             rsphere=6371200.,resolution='l',area_thresh=10000)

	# m = Basemap(width=12000000,height=8000000,
 #            resolution='l',projection='stere',\
 #            lat_ts=50,lat_0=lat_0,lon_0=lon_0)

	m = Basemap(width=8000000,height=5000000,
            resolution='l',projection='eqdc',\
            lat_1=50.,lat_2=60,lat_0=lat_0,lon_0=lon_0)
	

	# m = Basemap(projection='mill',\
	#             llcrnrlat=lllat,urcrnrlat=urlat,\
	#             llcrnrlon=lllon,urcrnrlon=urlon,resolution='l') # mill

	# m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            # llcrnrlon=-180,urcrnrlon=180,resolution='c')
	# draw coastlines, state and country boundaries, edge of map.
	m.drawcoastlines()
	m.drawstates()
	m.drawcountries()
	# draw parallels.
	# parallels = np.arange(0.,90,5.)
	parallels = np.arange(35.,80,10.)
	m.drawparallels(parallels,labels=[1,0,0,0],fontsize=fsize)
	# draw meridians
	meridians = np.arange(70.,180.,30.)
	# meridians = np.arange(70.,160.,10.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fsize)

	ny = data.shape[0]; nx = data.shape[1]
	lon_1d = np.array(np.arange(70.25,160,0.5))
	lons = np.repeat(lon_1d[np.newaxis,:],80,axis=0)
	lat_1d = np.array(np.arange(35.25,75,0.5))
	lats = np.repeat(lat_1d[:,np.newaxis],180,axis=1)

	# lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid. # the generated lons and lats are not right
	x, y = m(lons, lats) # compute map proj coordinates. 
	# draw filled contours.
	# clevs = [0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750]
	# cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)
	# cs = m.contourf(x,y,data,clevs,cmap=plt.cm.coolwarm) ## 
	clist =  [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
	colors = plt.cm.RdYlBu_r(clist)
	cmap1 = ListedColormap(colors)
	cs = m.contourf(x,y,data,clevs,cmap=plt.cm.RdYlBu_r,extend ='both') ## 
	# cs = m.contourf(x,y,data,clevs,cmap=cmap1,extend ='both') ## 
	# cs = m.contourf(x,y,data,clevs,cmap=plt.cm.Set3)
	# add colorbar.
	# cbar = m.colorbar(cs,location='bottom',pad="5%",extend='both')
	cbar = m.colorbar(cs,location='bottom',pad="10%",extend='both')
	# cbar.set_label('m')
	# add title
	cbar.ax.tick_params(labelsize = fsize)
	plt.tight_layout(pad=3,w_pad = 3)
	plt.title(out_title,fontsize= fsize)
	plt.show()
	fig.savefig(outfig)

	# plt.contour(X, Y, Z, colors='black');


def geo_grid_2_back4(data1,data2,data3,data4,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs1,clevs2,clevs3,clevs4,out_title,outfig):

	fig = plt.figure(figsize=(12,8))
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	m = Basemap(projection='cyl',\
	            llcrnrlat=lllat,urcrnrlat=urlat,\
	            llcrnrlon=lllon,urcrnrlon=urlon,resolution='l')

	m.drawcoastlines(color = 'grey')
	m.drawstates(color = 'grey')
	m.drawcountries(color = 'grey')

	parallels = np.arange(0.,90,10.)
	m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,color = 'grey')
	meridians = np.arange(0.,180.,10.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,color = 'grey')


	ny = data1.shape[0]; nx = data1.shape[1]
	lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
	x, y = m(lons, lats) # compute map proj coordinates.
	# clevs1 = np.arange(np.int(data2.min()),np.int(data2.max()),200)
	# clevs1 = np.arange(-100,2300,200)
	# pdb.set_trace()
	cs1 = m.contour(x,y,data1,clevs1,colors='b',linestyles='solid',linewidths = 1.5) # contour line 
	plt.clabel(cs1, inline=1, fontsize= 10, fmt = '%d')
	cs2 = m.contour(x,y,data2,clevs2,colors='r',linestyles='dashed',linewidths = 1.5) # contour line # colors= 'y'
	plt.clabel(cs2, inline=1, fontsize= 10, fmt = '%d')
	cs3 = m.contourf(x,y,data3,clevs3,cmap=plt.cm.coolwarm) # filled contour
	cs4 = m.contour(x,y,data4,clevs4,colors= 'k',linestyles='solid',linewidths = 0.8) # contour line
	plt.clabel(cs4, inline=1, fontsize= 8,fmt = '%d')

	cbar = m.colorbar(cs3,location='bottom',pad="5%")
	plt.title(out_title)
	plt.show()
	fig.savefig(outfig)

def geo_grid_2_back3(data1,data2,data3,lon_0,lat_0,lllon,lllat,urlon,urlat,clevs1,clevs2,clevs3,out_title,outfig):

	fig = plt.figure(figsize=(12,8))
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	m = Basemap(projection='cyl',\
	            llcrnrlat=lllat,urcrnrlat=urlat,\
	            llcrnrlon=lllon,urcrnrlon=urlon,resolution='l')

	m.drawcoastlines(color = 'grey')
	m.drawstates(color = 'grey')
	m.drawcountries(color = 'grey')

	parallels = np.arange(0.,90,10.)
	m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,color = 'grey')
	meridians = np.arange(0.,180.,10.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,color = 'grey')


	ny = data1.shape[0]; nx = data1.shape[1]
	lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
	x, y = m(lons, lats) # compute map proj coordinates.
	# clevs1 = np.arange(np.int(data2.min()),np.int(data2.max()),200)
	# clevs1 = np.arange(-100,2300,200)
	# pdb.set_trace()
	cs1 = m.contour(x,y,data1,clevs1,colors='b',linestyles='solid',linewidths = 1.5) # contour line 
	plt.clabel(cs1, inline=1, fontsize= 10, fmt = '%d')
	cs2 = m.contour(x,y,data2,clevs2,colors='r',linestyles='dashed',linewidths = 1.5) # contour line # colors= 'y'
	plt.clabel(cs2, inline=1, fontsize= 10, fmt = '%d')
	cs3 = m.contourf(x,y,data3,clevs3,cmap=plt.cm.coolwarm) # filled contour
	# cs4 = m.contour(x,y,data4,clevs4,colors= 'k',linestyles='solid',linewidths = 0.8) # contour line
	# plt.clabel(cs4, inline=1, fontsize= 8,fmt = '%d')

	cbar = m.colorbar(cs3,location='bottom',pad="5%")
	plt.title(out_title)
	plt.show()
	fig.savefig(outfig)



def geo_plot_point(lon,lat,cdata,sdata,out_title,out_fig):

	lllon = 70 + 0.25
		# lllat = 10 + 0.25
	lllat = 35 + 0.25
	urlon = 160 - 0.25
		# urlat = 55 - 0.25  
	urlat = 75 - 0.25
	lon_0 = (lllon+urlon)/2
	lat_0 = (lllat+urlat)/2
	# location and property
	# cdata -- value for color(length); sdata -- value for size (extent)
	fig = plt.figure(figsize=(12,8))
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	m = Basemap(projection='cyl',\
	            llcrnrlat=lllat,urcrnrlat=urlat,\
	            llcrnrlon=lllon,urcrnrlon=urlon,resolution='l') # mill

	m.drawcoastlines()
	m.drawstates()
	m.drawcountries()

	parallels = np.arange(0.,90,10.)
	m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
	meridians = np.arange(0.,180.,10.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

	m.scatter(lon, lat, latlon=True,
          c=cdata*10, s=(sdata)/10,
          cmap=plt.cm.coolwarm, alpha=0.5) # "Reds"

	plt.title(out_title)

	# create colorbar and legend
	plt.clim(0, 5)
	cax = plt.axes([0.92, 0.18, 0.02, 0.65])
	plt.colorbar(cax=cax)
	# plt.colorbar(label=r'p(blocking|heatwave)')
	# plt.colorbar(orientation= 'horizontal')
	# cax = plt.axes([0.85, 0.1, 0.075, 0.8])



	# make legend with dummy points
	for a in [10, 100, 500]: 
	    plt.scatter([], [], c='k', alpha=0.5, s=a/10,
	                label=str(a) + ' grids')
	plt.legend(scatterpoints=1, frameon=False,
	           labelspacing=1, bbox_to_anchor=(0.65, 0.22)) #loc='lower right',



	plt.show()
	fig.savefig(out_fig)



def plot_bar(data1,data2,out_title, outfig): 
	fig = plt.figure(figsize=(12,8))
	name_list = [str(day) for day in range(3,14)]
	num_list = data1
	num_list1 = data2
	x =list(range(len(num_list)))
	total_width, n = 0.8, 2
	width = total_width / n
 
	plt.bar(x, num_list, width=width, label='before_1996',fc = 'cornflowerblue')
	for i in range(len(x)):
		x[i] = x[i] + width
	plt.bar(x, num_list1, width=width, label='after_1996',tick_label = name_list,fc = 'salmon')
	plt.legend()
	plt.title(out_title)
	plt.show()
	fig.savefig(outfig)



def main():
	threshold = False
	data_threshold = 0.5
	vmax = 400
	vmin = 0
	# ***** gph_daily grid ********
	# data=sio.loadmat('summer_gph_daily_gridnum_day3.mat')['sum_gph_daily_gird']
	# outfig='summer_gph_daily_gridnum_day3.png'

	# data=sio.loadmat('summer_gph_daily_gird_day3_30_40.mat')['sum_gph_daily_gird']
	# outfig='summer_gph_daily_gird_day3_30_40.png'


	# ******** hwc daily grid **********
	# data=sio.loadmat('summer_hwc_daily_gridnum_day3.mat')['sum_hwc_daily_gird']
	# outfig='summer_hwc_daily_gridnum_day3.png'
	
	data=sio.loadmat('summer_hw_daily_gridnum_30_40.mat')['sum_hw_daily_gird']
	outfig='summer_hwc_daily_gridnum_day3_30_40.png'
	# ******* persentage **********
	# data=sio.loadmat('summer_percentage_of_hwandblo_in_hwedays_30_40.mat')['per_hw']
	# outfig='summer_percentage_of_hwandblo_in_hwedays_30_40_0.5.png'

	# data=sio.loadmat('summer_percentage_of_hwandblo_in_blodays_30_40.mat')['per_blo']
	# outfig='summer_percentage_of_hwandblo_in_blodays_30_40.png'

	# ******** blocking and hw days ************
	# data=sio.loadmat('summer_both_blocking_hw_daily_gridnum.mat')['gph_hw_sum']
	# outfig='summer_both_blocking_hw_daily_gridnum.png'
	# *********** persistence day of gph *************
	# data=sio.loadmat('summer_gph_perst_day3_day25.mat')['sum_gph_perst_all']
	# outfig='summer_gph_perst_day3_day25.png'

	# data=sio.loadmat('summer_gph_perst_all_day3_30_40.mat')['sum_gph_perst_all']
	# outfig='summer_gph_perst_all_day3_30_40.png'

	# ********  persistence day of blocking ***********
	# data=sio.loadmat('summer_hwc_perst_day3_day25.mat')['sum_hw_perst_all']
	# outfig='summer_hwc_perst_day3_day25_startday7.png'

	# ******** gph quantiel in heatwave************ 
	# data=sio.loadmat('gph_sum_quantile.mat')['gph_sum_quantile']
	# outfig='summer_gph_quantile_whole_summer.png'
	
	# # ******** gph quantile persentage in heatwave************ 
	# data=sio.loadmat('summer_blo_in_hwe_quantile_percentage.mat')['gph_sum_quantile_percentage']
	# outfig= 'summer_blo_in_hwe_quantile_percentage.png'
	
	# ******************* grid percentage **************** 
	# data=sio.loadmat('summer_grid_percentage_blo_in_hw.mat')['percentage_blo_in_hw']
	# data=data[:-20,:]
	# data[np.isnan(data)]=-0.1
	# data=np.flipud(data)
	# outfig= 'summer_grid_percentage_blo_in_hw.png'

	# data[:4,:]=0
	if threshold:
		data[np.where(data<data_threshold)]=0
	plot_grid(data,outfig,vmax,vmin)

# main()