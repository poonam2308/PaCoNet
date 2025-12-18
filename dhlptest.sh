# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json
#./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/color data/dhlp/pcw_test

# Step 2
# create masks for test data
#./src/dhlp//dataset/gen_mask.py data/dhlp/pcw_test/test data/dhlp/pcw_test/masks

# Step 3
#generate the predictions using the best checkpoint for the test data
#./src/dhlp/process_original.py ./src/dhlp/config/test.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth

# Step 4
# calculate sap, map
#./src/dhlp/eval-sAP.py ./outputs/results_test3
#./src/dhlp/eval-mAP.py ./outputs/results_test3
#./src/dhlp/demo.py -d 0 ./src/dhlp/config/test.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth ./outputs/syns/p_test/

#step 5 to get the mean and mean offsets

#./src/dhlp//process_offset_dist.py ./src/dhlp/config/test.yaml ./outputs/chkpt/dhlp/checkpoint_best_c.pth
#./src/dhlp//process_offset_sing.py ./src/dhlp/config/test.yaml ./outputs/chkpt/dhlp/checkpoint_best_c.pth
#----------------------------------- color no unet-----------------------------------------------------------------------
#./src/dhlp/process_offset_dist.py ./src/dhlp/config/noisedTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_nc.pth
#./src/dhlp/process_offset_sing.py ./src/dhlp/config/noisedTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_nc.pth
#----------------------------------- cluster---------------------------------------------------------------------------
#./src/dhlp/process_offset_dist.py ./src/dhlp/config/testCluster.yaml ./outputs/chkpt/dhlp/checkpoint_best_cls.pth
#./src/dhlp/process_offset_sing.py ./src/dhlp/config/testCluster.yaml ./outputs/chkpt/dhlp/checkpoint_best_cls.pth

#----------------------------------- cluster no unet----------------------------------------------------------------------
#./src/dhlp/process_offset_dist.py ./src/dhlp/config/noisedClusterTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_ncls.pth
#./src/dhlp/process_offset_sing.py ./src/dhlp/config/noisedClusterTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_ncls.pth


#------------------------------------------------------------------------------------------------------------------------


#./src/dhlp//process_offset_sing_hungarian.py ./src/dhlp/config/testCluster.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth



# ------------ -------------------------------
#--------------------------------------------
# color unet 1

#./src/dhlp//process_offset_sing_try.py ./src/dhlp/config/test.yaml ./outputs/chkpt/dhlp/checkpoint_best_c.pth

# color no unet  2
#./src/dhlp//process_offset_sing_try.py ./src/dhlp/config/noisedTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_nc.pth

# cluster  unet 3
#./src/dhlp//process_offset_sing_try.py ./src/dhlp/config/testCluster.yaml ./outputs/chkpt/dhlp/checkpoint_best_cls.pth

# cluster no unet 4
#./src/dhlp//process_offset_sing_try.py ./src/dhlp/config/noisedClusterTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_ncls.pth


# ------------ -------------------------------
#--------------------------------------------

#./src/dhlp//process_offset_dist.py ./src/dhlp/config/testCluster.yaml ./outputs/chkpt/dhlp/checkpoint_best.pth

#
#./src/dhlp/process_offset_sing_mae.py ./src/dhlp/config/test.yaml ./outputs/chkpt/dhlp/checkpoint_best_c.pth


#./src/dhlp/process_offset_sing_mae1.py  ./src/dhlp/config/test.yaml ./outputs/chkpt/dhlp/checkpoint_best_c.pth


#Final Average Offset Errors (Nearest Junction Based):
#Avg Mean Offset: 24.49
#Avg Lower Offset: -10.44
#Avg Upper Offset: 59.41


####------------------------##################################
# create the dataset as dhlp format for the crops
#
#./src/dhlp/dataset/wireframe_all_test.py data/synthetic_plots/multi_cat/testing/m_crops data/pcw_crops_test

#./src/dhlp//dataset/gen_mask.py data/pcw_crops_test/test data/pcw_crops_test/masks

#-----------------noisy inputs of test data (without unet)
# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json

#./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/color data/dhlp/pcw_ntest

# Step 2
# create masks for test data
#./src/dhlp//dataset/gen_mask.py data/dhlp/pcw_ntest/test data/dhlp/pcw_ntest/masks

#%% ------------ noisy cluster images ----------
# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json

#./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/cluster data/dhlp/pcw_ntest_cls

# Step 2
# create masks for test data
#./src/dhlp//dataset/gen_mask.py data/dhlp/pcw_ntest_cls/test data/dhlp/pcw_ntest_cls/masks


#%% ------------ denoised unet cluster  ----------
# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json
#
#./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/cluster data/dhlp/pcw_test_cls

# Step 2
# create masks for test data
#./src/dhlp//dataset/gen_mask.py data/dhlp/pcw_test_cls/test data/dhlp/pcw_test_cls/masks

