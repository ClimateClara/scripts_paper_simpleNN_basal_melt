import pandas as pd
import xarray as xr
import numpy as np

# Formatting the data

# the domain stuff
def cut_domain_stereo(var_to_cut, map_lim_x, map_lim_y):
    var_cutted = var_to_cut.sel(x=var_to_cut.x.where(in_range(var_to_cut.x,map_lim_x),drop=True), y=var_to_cut.y.where(in_range(var_to_cut.y,map_lim_y),drop=True))
    return var_cutted

def in_range(in_xy,txy):
    return ((in_xy >= min(txy)) & (in_xy < max(txy)))

def prepare_input_df_1_year(TS_prof_isf, melt_rate_isf, yy, max_front_depth, geometry_2D_isf, only_1_yr_tot=False):

    # take profile of T and S for that ice shelf and only as deep as the deepest entrance point
    TS_isf_tt = TS_prof_isf.sel(time=yy).where(TS_prof_isf.depth < max_front_depth, drop=True).drop('profile_domain').drop('Nisf')
    if only_1_yr_tot:
        TS_isf_tt = TS_isf_tt.drop('time')
    # T and S profiles to dataframe
    TS_isf_df = TS_isf_tt.to_dataframe()
    
    # Geometry info to dataframe
    length_df = len(geometry_2D_isf.x)*len(geometry_2D_isf.y)
    geo_df = geometry_2D_isf.to_dataframe()
    
    # Write the T and S profiles in different columns for this year (all rows the same)
    
    T_list = [ ]
    S_list = [ ]
    depth_list = [ ]
    for ii in range(len(TS_isf_tt.depth)):
        T_list.append('T_'+str(ii).zfill(3))
        S_list.append('S_'+str(ii).zfill(3))
        depth_list.append('d_'+str(ii).zfill(3))
    
    for nn in range(length_df):
        for ii,icol in enumerate(T_list):
            geo_df[icol] = TS_isf_df['theta_ocean'].values[ii]
        for ii,icol in enumerate(S_list):    
            geo_df[icol] = TS_isf_df['salinity_ocean'].values[ii]
    
    # write out melt rate for that year (target)
    melt_rate_isf_tt = melt_rate_isf.sel(time=yy)
    # transform to dataframe
    melt_df = melt_rate_isf_tt.drop('longitude').drop('latitude').to_dataframe()#.drop(['mapping'],axis=1)#.reset_index()
    if only_1_yr_tot:
        melt_df = melt_df.drop(['time'],axis=1)
    
    # merge properties and melt
    merged_df = pd.merge(geo_df,melt_df,how='left',on=['x','y'])

    # remove rows where there are nans
    clean_df = merged_df.dropna()
    # remove the x and y axis
    # clean_df = clean_df.drop(['x'], axis=1).drop(['y'], axis=1)
    return clean_df, T_list, S_list


def weighted_mean(data, dims, weights):
    weight_sum = weights.sum(dim=dims) # to avoid dividing by zero
    return (data*weights).sum(dim=dims)/weight_sum.where(weight_sum != 0)

def create_stacked_mask(isfmask_2D, nisf_list, dims_to_stack, new_dim):
    # create stacked indices to select the different ice shelves
    # based on https://xarray.pydata.org/en/stable/indexing.html#more-advanced-indexing
    stacked_mask = isfmask_2D.stack(z=(dims_to_stack))
    
    new_coord_mask = stacked_mask.z.where(stacked_mask==nisf_list).dropna(how='all',dim='z')  
    new_coord_mask = new_coord_mask.rename({'z': new_dim})
    
    return new_coord_mask

def choose_isf(var, isf_stacked_mask, kisf):
    # choose a given ice shelf based on stacked coordinate
    return var.stack(mask_coord=['y','x']).sel(mask_coord=isf_stacked_mask.sel(Nisf=kisf).dropna('mask_coord'))

def check_slope_one_dimension(input_da, shifted_plus, shifted_minus, dx):

    """
    Compute the basal slope at each point.
        
    Parameters
    ----------
    input_da : xr.DataArray
        Array where slope needs to be checked. For example: ice draft.
    shifted_plus : xr.DataArray
        Shifted version (positive direction) of input_da.
    shifted_minus : xr.DataArray
        Shifted version (negative direction) of input_da.
    dx : float
        Step in the coordinate along which input_da was shifted

    Returns
    -------
    slope: xr.DataArray
        slope along that coordinate, is 0 if nan
    """
    
    # check the direction in both dim directions
    slope_both = (shifted_minus - shifted_plus) / np.sqrt((2 * dx) ** 2)
    # if x+1 is nan, only take x - (x-1)
    slope_right = (input_da - shifted_plus) / np.sqrt(dx ** 2)
    # if x-1 is nan, only take x+1 - x
    slope_left = (shifted_minus - input_da) / np.sqrt(dx ** 2)
    # combine all of the above
    slope = slope_both.combine_first(slope_right).combine_first(slope_left)
    # set rest to 0
    slope = slope.where(np.isfinite(slope), 0)
    return slope

def check_slope_one_dimension_latlon(input_da, shifted_plus, shifted_minus, latlon, shifted_latlon_plus, shifted_latlon_minus):

    """
    Compute the zonal and meridional slope in x and y direction.
        
    Parameters
    ----------
    input_da : xr.DataArray
        Array where slope needs to be checked. For example: ice draft.
    shifted_plus : xr.DataArray
        Shifted version (positive direction) of input_da.
    shifted_minus : xr.DataArray
        Shifted version (negative direction) of input_da.
    latlon : xr.DataArray
        Latitude or longitude, depending on which of zonal or meridional is needed.
    shifted_latlon_plus : xr.DataArray
        Shifted version (positive x or y direction) of latlon.
    shifted_latlon_minus : xr.DataArray
        Shifted version (negative x or y direction) of latlon.

    Returns
    -------
    slope: xr.DataArray
        slope along that coordinate, is 0 if nan
    """
    
    # check the direction in both dim directions
    slope_both = (shifted_minus - shifted_plus) / (shifted_latlon_minus - shifted_latlon_plus)
    # if x+1 is nan, only take x - (x-1)
    slope_right = (input_da - shifted_plus) / (latlon - shifted_latlon_plus)
    # if x-1 is nan, only take x+1 - x
    slope_left = (shifted_minus - input_da) / (shifted_latlon_minus - latlon)
    # combine all of the above
    slope = slope_both.combine_first(slope_right).combine_first(slope_left)
    # set rest to 0
    slope = slope.where(np.isfinite(slope), 0)
    return slope


def compute_alpha_appenB(kisf, plume_var_of_int, ice_draft_neg, dx, dy):   

    """
    Compute alphas like in Appendix B of Favier et al., 2019 TCDiscussions. - copied from plume_functions from multimelt
    
    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` and ``'dIF'``
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards.
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
        
    Returns
    -------
    go_back_to_whole_grid_local_alpha: xr.DataArray
        Local slope angle in rad for each point.
    """
    
    # cut out the ice shelf of interest
    draft_isf = ice_draft_neg.where(plume_var_of_int['ISF_mask'] == kisf, drop=True)

    shiftedx_minus = draft_isf.shift(x=-1)
    shiftedx_plus = draft_isf.shift(x=1)
    xslope = check_slope_one_dimension(draft_isf, shiftedx_plus, shiftedx_minus, dx)

    shiftedy_minus = draft_isf.shift(y=-1)
    shiftedy_plus = draft_isf.shift(y=1)
    yslope = check_slope_one_dimension(draft_isf, shiftedy_plus, shiftedy_minus, dy)

    dIF_isf = plume_var_of_int['dIF'].where(plume_var_of_int['ISF_mask'] == kisf)
    dIF_isf_corr = dIF_isf.where(dIF_isf/2500 < 1,1) #check again with Nico, if I understood it right (MIN to avoid strong frontal slope)

    local_alpha = np.arctan(np.sqrt(xslope ** 2 + yslope ** 2)) * dIF_isf_corr

    go_back_to_whole_grid_local_alpha = local_alpha.reindex_like(plume_var_of_int['ISF_mask'])

    return go_back_to_whole_grid_local_alpha

def slope_in_x_and_y_dir(kisf, plume_var_of_int, ice_draft_neg, dx, dy):


    """
    Compute slope in x and slope in y direction
    
    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` 
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards. Or other variable on which we want the slope (e.g. bathymetry)
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
        
    Returns
    -------
    xslope_whole_grid: xr.DataArray
        Slope (dvar/dx) in x-direction for each point.
    yslope_whole_grid: xr.DataArray
        Slope (dvar/dy) in y-direction for each point.
    """
    
    # cut out the ice shelf of interest
    draft_isf = ice_draft_neg.where(plume_var_of_int['ISF_mask'] == kisf, drop=True)
    
    shiftedx_minus = draft_isf.shift(x=-1)
    shiftedx_plus = draft_isf.shift(x=1)
    xslope = check_slope_one_dimension(draft_isf, shiftedx_plus, shiftedx_minus, dx)

    shiftedy_minus = draft_isf.shift(y=-1)
    shiftedy_plus = draft_isf.shift(y=1)
    yslope = check_slope_one_dimension(draft_isf, shiftedy_plus, shiftedy_minus, dy)
    
    xslope_whole_grid = xslope.reindex_like(plume_var_of_int['ISF_mask'])
    yslope_whole_grid = yslope.reindex_like(plume_var_of_int['ISF_mask'])
    
    return xslope_whole_grid, yslope_whole_grid

def slope_in_lon_and_lat_dir(kisf, plume_var_of_int, ice_draft_neg, var):
    

    """
    Compute lon and lat slope in x and slope in y direction
    
    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` , ``'longitude'``, ``'latitude'``
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards. Or other variable on which we want the slope (e.g. bathymetry)
    var: string
        Name of the variable contained in ice_draft_neg
        
    Returns
    -------
    xslope_lat_whole_grid: xr.DataArray
        Slope (dvar/dlat) in x-direction for each point.
    xslope_lon_whole_grid: xr.DataArray
        Slope (dvar/dlon) in x-direction for each point.
    yslope_lat_whole_grid: xr.DataArray
        Slope (dvar/dlat) in y-direction for each point.
    yslope_lon_whole_grid: xr.DataArray
        Slope (dvar/dlon) in y-direction for each point.
        
    """
        
    
    # cut out the ice shelf of interest
    #print('dfmt1')
    draft_isf = ice_draft_neg.where(plume_var_of_int['ISF_mask'] == kisf, drop=True)
    #print('dfmt2')
    lat = plume_var_of_int['latitude'].where(plume_var_of_int['ISF_mask'] == kisf, drop=True).drop('latitude').drop('longitude')
    #print('dfmt3')
    lon = plume_var_of_int['longitude'].where(plume_var_of_int['ISF_mask'] == kisf, drop=True).drop('latitude').drop('longitude')
    #print('dfmt4')
    shift_vars = xr.merge([draft_isf.drop('latitude').drop('longitude'),lat,lon])
    
    #print('dfmt5')
    shift_vars_x_minus = shift_vars.shift(x=-1)
    shift_vars_x_plus = shift_vars.shift(x=1)
    shift_vars_y_minus = shift_vars.shift(y=-1)
    shift_vars_y_plus = shift_vars.shift(y=1)
    
    #print('dfmt6')
    for ccoord in ['longitude','latitude']:
    
        shift_vars['xslope_'+ccoord] = check_slope_one_dimension_latlon(shift_vars[var], shift_vars_x_plus[var], shift_vars_x_minus[var], 
                                                  shift_vars[ccoord], shift_vars_x_plus[ccoord], shift_vars_x_minus[ccoord])
        shift_vars['yslope_'+ccoord] = check_slope_one_dimension_latlon(shift_vars[var], shift_vars_y_plus[var], shift_vars_y_minus[var], 
                                                  shift_vars[ccoord], shift_vars_y_plus[ccoord], shift_vars_y_minus[ccoord])
    
    #print('dfmt7')
    xslope_lat_whole_grid = shift_vars['xslope_latitude'].reindex_like(plume_var_of_int['ISF_mask'])
    xslope_lon_whole_grid = shift_vars['xslope_longitude'].reindex_like(plume_var_of_int['ISF_mask'])

    yslope_lat_whole_grid = shift_vars['yslope_latitude'].reindex_like(plume_var_of_int['ISF_mask'])
    yslope_lon_whole_grid = shift_vars['yslope_longitude'].reindex_like(plume_var_of_int['ISF_mask']) 
    
    return xslope_lat_whole_grid, xslope_lon_whole_grid, yslope_lat_whole_grid, yslope_lon_whole_grid

def slope_zonal_merid(kisf, plume_var_of_int, ice_draft_neg, dx, dy):
    

    """
    Compute lon and lat slope in x and slope in y direction
    
    Parameters
    ----------
    kisf : int
        ID of the ice shelf of interest
    plume_var_of_int : xr.Dataset
        Dataset containing ``'ISF_mask'`` , ``'longitude'``, ``'latitude'``
    ice_draft_neg : xr.DataArray
        Ice draft depth in m. Negative downwards. Or other variable on which we want the slope (e.g. bathymetry)
    dx : float
        Grid spacing in the x-direction
    dy : float
        Grid spacing in the y-direction
    
    Returns
    -------
    merid_slope_whole_grid: xr.DataArray
        Slope (dvar/dlon) for each point.
    zonal_slope_whole_grid: xr.DataArray
        Slope (dvar/dlat) for each point.
        
    """
    
    # cut out the ice shelf of interest
    draft_isf = ice_draft_neg.where(plume_var_of_int['ISF_mask'] == kisf, drop=True)
    lon_isf = plume_var_of_int['longitude'].where(plume_var_of_int['ISF_mask'] == kisf, drop=True)
    
    shiftedx_minus = draft_isf.shift(x=-1)
    shiftedx_plus = draft_isf.shift(x=1)
    xslope = check_slope_one_dimension(draft_isf, shiftedx_plus, shiftedx_minus, dx)

    shiftedy_minus = draft_isf.shift(y=-1)
    shiftedy_plus = draft_isf.shift(y=1)
    yslope = check_slope_one_dimension(draft_isf, shiftedy_plus, shiftedy_minus, dy)
        
    dr = np.pi/180.
    #deltaphi = 90. - GEddxx !Geddxx= 90° in my config
    deltaphi = 0

    phi = (-1)*(lon_isf+deltaphi)*dr #to turn positively
    cphi = np.cos( phi )
    sphi = np.sin( phi )
    #
    ux = cphi*xslope.values + sphi*(-1)*yslope.values #because the y-axis is in the other direction
    vy = - sphi*xslope.values + cphi*(-1)*yslope.values #because the y-axis is in the other direction
    u_lon = xr.DataArray(ux, coords=lon_isf.coords).transpose('y','x').rename('slope_lon')
    v_lat = xr.DataArray(vy, coords=lon_isf.coords).transpose('y','x').rename('slope_lat')
    
    merid_slope_whole_grid = u_lon.reindex_like(plume_var_of_int['ISF_mask'])
    zonal_slope_whole_grid = v_lat.reindex_like(plume_var_of_int['ISF_mask'])
    
    return merid_slope_whole_grid, zonal_slope_whole_grid

def print_shape_xr_ds(ds):
    
    """
    Write out dimensions of an xarray Dataset (index, variables)
    
    Parameters
    ----------
    ds: xr.Dataset
        Dataset with a given amount of variables and the dimension ``index``
        
    Returns
    -------
    Tuple with the length of index and number of variables
        
    """
    
    return (len(ds.index),len(ds.data_vars.keys()))