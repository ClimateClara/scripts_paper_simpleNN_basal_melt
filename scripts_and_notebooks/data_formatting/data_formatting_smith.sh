#!/bin/bash

#########
#This script is to format the raw NEMO data(from the Smith runs) for further use 
#It extracts the interesting variables and adds the right grid information needed to regrid it in the next step
########

homepath=/bettik/burgardc

#### name of NEMO run
# nemo_run=bf663
nemo_run=bi646

################# DECLARE THE PATHS ##############################
path1=$homepath/DATA/NN_PARAM/raw/SMITH_DATA
path2=$homepath/DATA/NN_PARAM/interim/SMITH_"$nemo_run"
path5=$homepath/DATA/NN_PARAM/raw
###################################################################



###### VARIABLES
ncks -d y,0,500 $path5/grid_eORCA025_CDO_Fabien.nc -o $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc

echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc $path2/3D_variables_of_interest_allyy.nc 

for var in {thetao,so}
do
echo $var
echo 'ncks > extract variable from gridT' $var
ncks -O -C -v $var $path1/"$nemo_run"_1y_grid-T_AVG.nc $path2/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/$var.nc $path2/3D_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/3D_variables_of_interest_allyy.nc
\rm $path2/$var.nc
done

echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc $path2/2D_variables_of_interest_allyy.nc 

var=sowflisf
echo $var
echo 'ncks > extract variable from gridT' $var
ncks -O -C -v $var $path1/"$nemo_run"_1y_isf-T_AVG.nc $path2/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/$var.nc $path2/2D_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/2D_variables_of_interest_allyy.nc
\rm $path2/$var.nc

var=tos
echo $var
echo 'ncks > extract variable from gridT' $var
ncks -O -C -v $var $path1/"$nemo_run"_1y_grid-T_AVG.nc $path2/$var.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/$var.nc $path2/2D_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/2D_variables_of_interest_allyy.nc
\rm $path2/$var.nc


echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc $path2/3D_depth_coord.nc 

var=gdept_0
echo $var
echo 'ncks > extract variable from mesh_mask' $var
ncks -O -C -v $var $path1/nemo_"$nemo_run"c_18801201_mesh_mask.nc $path2/$var.nc # add c if bi646, o if bf663
echo 'cut region of interest'
ncks -d y,0,500 $path2/$var.nc $path2/"$var"_cut.nc
echo 'ncks > put variable in file' $var
ncks -A -C -v $var $path2/"$var"_cut.nc $path2/3D_depth_coord.nc 
echo 'ncatted > put coords to lon lat' $var
ncatted -a coordinates,$var,m,c,"lon lat" $path2/3D_depth_coord.nc 
\rm $path2/$var.nc
\rm $path2/"$var"_cut.nc


###### MASK

echo 'cp > create gridded file'
cp $path5/grid_eORCA025_CDO_Fabien_southofaround50.nc $path2/mask_variables_of_interest_allyy.nc

# write out the ice-shelf draft and the bathymetry 
for var in {isf_draft,Bathymetry_isf} 
do
echo $var
echo 'ncks > extract variable'
ncks -O -C -v $var $path1/bf663c_YYYY1201_bathymetry-isf.nc  $path2/$var.nc # TO CHANGE WHEN DOING BF663 bf663c_YYYY1201_bathymetry-isf.nc #bi646c_YYYY1201_bathymetry-isf.nc
echo 'ncks > put variable in file'
ncks -A -C -v $var $path2/$var.nc $path2/mask_variables_of_interest_allyy.nc
echo 'ncatted > put coords to lon lat'
ncatted -a coordinates,$var,m,c,"lon lat" $path2/mask_variables_of_interest_allyy.nc
done

cdo setgrid,$path2/2D_variables_of_interest_allyy.nc $path2/mask_variables_of_interest_allyy.nc $path2/mask_variables_of_interest_allyy_setgrid.nc
cdo sellonlatbox,0,360,-90,-50 $path2/mask_variables_of_interest_allyy_setgrid.nc $path2/mask_variables_of_interest_allyy_Ant.nc
cdo sellonlatbox,0,360,-90,-50 $path2/3D_variables_of_interest_allyy.nc $path2/3D_variables_of_interest_allyy_Ant.nc
cdo sellonlatbox,0,360,-90,-50 $path2/2D_variables_of_interest_allyy.nc $path2/2D_variables_of_interest_allyy_Ant.nc
cdo setgrid,$path2/2D_variables_of_interest_allyy.nc $path2/3D_depth_coord.nc $path2/3D_depth_coord_setgrid.nc
cdo sellonlatbox,0,360,-90,-50 $path2/3D_depth_coord_setgrid.nc $path2/3D_depth_coord_Ant.nc 


#### PREPARE LAND SEA MASK


cdo eqc,0 -sellevidx,1 -selvar,so $path2/3D_variables_of_interest_allyy_Ant.nc $path2/ocean0.nc # where there is no open ocean => 1
cdo eqc,0 -vertmax -selvar,so $path2/3D_variables_of_interest_allyy_Ant.nc $path2/ground1.nc # where there is no ocean or floating ice shelf => 1
cdo add $path2/ground1.nc $path2/ocean0.nc $path2/lsmask_012.nc


#### SET LAND TO NAN AND NOT 0 IN TEMPERATURE AND SALINITY

cdo gtc,0 -selvar,so $path2/3D_variables_of_interest_allyy_Ant.nc $path2/mask_for_ocean_through_salinity.nc
cdo ifthen -selvar,so $path2/mask_for_ocean_through_salinity.nc -selvar,so $path2/3D_variables_of_interest_allyy_Ant.nc $path2/salinity_allyy_Ant_withNaN.nc
cdo ifthen -selvar,so $path2/mask_for_ocean_through_salinity.nc -selvar,thetao $path2/3D_variables_of_interest_allyy_Ant.nc $path2/theta_allyy_Ant_withNaN.nc
cdo merge -selvar,thetao $path2/theta_allyy_Ant_withNaN.nc -selvar,so $path2/salinity_allyy_Ant_withNaN.nc $path2/TandS_allyy_Ant_withNaN.nc

cdo settaxis,1970-01-01,12:00:00,1year $path2/TandS_allyy_Ant_withNaN.nc $path2/TandS_allyy_Ant_withNaN_timeaxis.nc
cdo splityear $path2/TandS_allyy_Ant_withNaN_timeaxis.nc $path2/TandS_allyy_Ant_withNaN_

cdo settaxis,1970-01-01,12:00:00,1year $path2/mask_variables_of_interest_allyy_Ant.nc $path2/mask_variables_of_interest_allyy_Ant_timeaxis.nc
cdo settaxis,1970-01-01,12:00:00,1year $path2/lsmask_012.nc $path2/lsmask_012_timeaxis.nc

### WHEN HERE, CONTINUE WITH custom_lsmask_Smith.ipynb