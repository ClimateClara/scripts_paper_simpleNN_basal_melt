"""
Created on Mon Mar 27 17:37 2023

Apply the ensemble of NN to Smith data and only take ensemble mean
=> only 2D

Author: @claraburgard

"""

import numpy as np
import xarray as xr
import pandas as pd
import glob
import matplotlib as mpl
import seaborn as sns
import datetime
import time
import os,sys

import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout

from basal_melt_neural_networks.constants import *
import basal_melt_neural_networks.diagnostic_functions as diag
import basal_melt_neural_networks.data_formatting as dfmt
import basal_melt_neural_networks.postprocessing_functions as pp
from basal_melt_param.constants import *

#### INPUT STUFF

mod_size = str(sys.argv[1]) #'mini', 'xsmall96', 'small', 'medium', 'large', 'extra_large'
TS_opt = str(sys.argv[2]) # extrap, whole, thermocline
norm_method = str(sys.argv[3]) # std, interquart, minmax
exp_name = str(sys.argv[4])
nemo_run = str(sys.argv[5])

tblock_dim = range(1980, 1980 + 60)

##### READ IN DATA

inputpath_data_nn = '/bettik/burgardc/DATA/NN_PARAM/interim/INPUT_DATA/'

if TS_opt == 'extrap':
    inputpath_CVinput = inputpath_data_nn+'EXTRAPOLATED_ISFDRAFT_CHUNKS/'
    inputpath_csv = inputpath_data_nn+'SMITH_'+nemo_run+'_EXTRAPDRAFT_CHUNKS/'
elif TS_opt == 'whole':
    inputpath_CVinput = inputpath_data_nn+'WHOLE_PROF_CHUNKS_CV/'
    path_orig_data = inputpath_data_nn+'WHOLE_PROF_CHUNKS/'
elif TS_opt == 'thermocline':
    inputpath_CVinput = inputpath_data_nn+'THERMOCLINE_CHUNKS_CV/'
    
##### DECIDE INPUT
    
#input_vars = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat','isfdraft_conc','theta_in','salinity_in','u_tide']
if exp_name == 'onlyTSdraft':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in']
elif exp_name == 'TSdraftbotandiceddandwcd':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','water_col_depth','theta_bot','salinity_bot']
elif exp_name == 'TSdraftbotandiceddandwcdreldGL':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','water_col_depth','theta_bot','salinity_bot','rel_dGL']
elif exp_name == 'onlyTSdraftandslope':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','slope_ice_lon','slope_ice_lat']
elif exp_name == 'onlyTSdraft2':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in']
elif exp_name == 'TSTfdGLdIFwcd':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','dGL','dIF','slope_ice_lon','slope_ice_lat','water_col_depth']
elif exp_name == 'TSdraftslopereldGL':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','slope_ice_lon','slope_ice_lat','rel_dGL']
elif exp_name == 'allbutconstants':
    input_vars = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat',
                'isfdraft_conc','theta_in','salinity_in','u_tide']
elif exp_name == 'newbasic':
    input_vars = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat',
                'theta_in','salinity_in']
elif exp_name == 'newbasic2':
    input_vars = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat',
                'theta_in','salinity_in','T_mean', 'S_mean', 'T_std', 'S_std']
    
map_lim = [-3000000,3000000]
inputpath_data='/bettik/burgardc/DATA/NN_PARAM/interim/SMITH_'+nemo_run+'/'
file_other = xr.open_dataset(inputpath_data+'corrected_draft_bathy_isf.nc')#, chunks={'x': chunk_size, 'y': chunk_size})
file_other_cut = dfmt.cut_domain_stereo(file_other, map_lim, map_lim)
file_conc = xr.open_dataset(inputpath_data+'isfdraft_conc_Ant_stereo.nc')
file_conc_cut = dfmt.cut_domain_stereo(file_conc, map_lim, map_lim)

### use any model from CV over time
outputpath_melt_nn = '/bettik/burgardc/DATA/NN_PARAM/processed/MELT_RATE/SMITH_'+nemo_run+'/'
path_model = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/experiments/WHOLE/'

for tt in tblock_dim:
    print(tt)
    
    file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask = pp.read_input_evalmetrics_NN_yy(nemo_run, tt, file_conc_cut, file_other_cut)
    
    res_1D_list = []
    res_2D_all = None
    for kisf in file_isf.Nisf.values: 

        norm_metrics_file = xr.open_dataset(inputpath_CVinput + 'metrics_norm_wholedataset.nc')
        #norm_metrics_file_addvar1 = xr.open_dataset(inputpath_CVinput + 'metrics_norm_addvar1_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
        #norm_metrics_file_addvar1 = norm_metrics_file_addvar1.drop('salinity_in')
        #norm_metrics_file = xr.merge([norm_metrics_file_orig,norm_metrics_file_addvar1])
        norm_metrics = norm_metrics_file.sel(norm_method=norm_method).drop('norm_method').to_dataframe()

        df_nrun = pd.read_csv(inputpath_csv + 'dataframe_input_isf'+str(kisf).zfill(3)+'_'+str(tt)+'_'+nemo_run+'.csv',index_col=[0,1])
        #df_nrun_addvar1 = pd.read_csv(path_orig_data + 'dataframe_addvar1_isf'+str(kisf).zfill(3)+'_'+str(tblock_out).zfill(3)+'.csv',index_col=[0,1,2])
        #df_nrun_addvar1 = df_nrun_addvar1.drop(['salinity_in'], axis=1)
        #df_nrun = pd.concat([df_nrun_orig,df_nrun_addvar1],join = 'outer', axis = 1)
        
        ens_res2D_list = []
        for seed_nb in range(1,11):
            model = keras.models.load_model(path_model + 'model_nn_'+mod_size+'_'+exp_name+'_wholedataset_'+str(seed_nb).zfill(2)+'_TS'+TS_opt+'_norm'+norm_method+'.h5')

            res_2D = pp.apply_NN_results_2D_1isf_1tblock(file_isf, norm_metrics, df_nrun, model, input_vars)
        
            ens_res2D_list.append(res_2D.assign_coords({'seed_nb': seed_nb}))
        
        xr_ens_res2D = xr.concat(ens_res2D_list, dim='seed_nb')
        xr_ensmean_res2D = xr_ens_res2D.mean('seed_nb')
        #print('2D ready!')

        if box_charac_2D and box_charac_1D:
            box_loc_config2 = box_charac_2D['box_location'].sel(box_nb_tot=box_charac_1D['nD_config'].sel(config=2))
            box1 = box_loc_config2.where(box_loc_config2==1).isel(Nisf=1).drop('Nisf')

        geometry_isf_2D = dfmt.choose_isf(geometry_info_2D,isf_stack_mask, kisf)
        melt_rate_2D_isf_m_per_y = dfmt.choose_isf(xr_ensmean_res2D,isf_stack_mask, kisf)

        melt_rate_1D_isf_Gt_per_y = (melt_rate_2D_isf_m_per_y * geometry_isf_2D['grid_cell_area_weighted']).sum(dim=['mask_coord']) * rho_i / 10**12

        box_loc_config_stacked = dfmt.choose_isf(box1, isf_stack_mask, kisf)
        param_melt_2D_box1_isf = melt_rate_2D_isf_m_per_y.where(np.isfinite(box_loc_config_stacked))

        melt_rate_1D_isf_myr_box1_mean = dfmt.weighted_mean(param_melt_2D_box1_isf,['mask_coord'], geometry_isf_2D['isfdraft_conc'])     

        out_1D = xr.concat([melt_rate_1D_isf_Gt_per_y, melt_rate_1D_isf_myr_box1_mean], dim='metrics').assign_coords({'metrics': ['Gt','box1']})
        res_1D_list.append(out_1D)
        
        if res_2D_all is None:
            res_2D_all = xr_ensmean_res2D
        else:
            res_2D_all = res_2D_all.combine_first(xr_ensmean_res2D)
        #print('1D ready!')
    
    res_2D_all.to_netcdf(outputpath_melt_nn + 'evalmetrics_2D_'+mod_size+'_'+exp_name+'_ensmean_'+TS_opt+'_norm'+norm_method+'_'+str(tt)+'_'+nemo_run+'.nc')
    print('2D written to file')
    
    res_1D_all = xr.concat(res_1D_list, dim='Nisf')
    res_1D_all.to_netcdf(outputpath_melt_nn + 'evalmetrics_1D_'+mod_size+'_'+exp_name+'_ensmean_'+TS_opt+'_norm'+norm_method+'_'+str(tt)+'_'+nemo_run+'.nc')
    print('1D written to file')
    
    
#    if (tblock_out == max(tblock_dim)):
#        print('I AM SAVING RESULTS FOR NEMO RUN '+nemo_run_old)
#        res_all_CV = xr.concat(res_all_list, dim='time')
#        res_all_CV.to_netcdf(outputpath_melt_nn + 'evalmetrics_2D_CV_'+mod_size+'_'+exp_name+'_'+TS_opt+'_norm'+norm_method+'_'+nemo_run_old+'.nc')