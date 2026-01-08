
# ------------ -------------------------------
#--------------------------------------------
# color unet 1

#./src/dhlp/process_offset_sing_try_crop.py ./src/dhlp/config/test.yaml ./outputs/chkpt/dhlp/checkpoint_best_c.pth

# color no unet  2
#./src/dhlp/process_offset_sing_try_crop.py ./src/dhlp/config/noisedTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_nc.pth

# cluster  unet 3
#./src/dhlp/process_offset_sing_try_crop.py ./src/dhlp/config/testCluster.yaml ./outputs/chkpt/dhlp/checkpoint_best_cls.pth

# cluster no unet 4
#./src/dhlp/process_offset_sing_try_crop.py ./src/dhlp/config/noisedClusterTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_ncls.pth


#-----------------------------------------crops---------------------------------------------------------------------------

./src/dhlp/process_offset_sing.py ./src/dhlp/config/cropsTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_crop.pth

#./src/dhlp/process_offset_cat_dist.py ./src/dhlp/config/cropsTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_crop.pth

#------------------------------------------------------------------------------------------------------------------------

