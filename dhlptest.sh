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

#Evaluating ./outputs/results_test
#  27.4 | 38.4
#Evaluating ./outputs/results_test1
#  30.9 | 44.0
#Evaluating ./outputs/results_test2
#  30.7 | 44.2
#Evaluating ./outputs/results_test3
#  33.6 | 48.5
#
#./src/dhlp/demo.py -d 0 ./src/dhlp/config/test.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth ./outputs/syns/p_test/
#


#step 5 to get the mean and mean offsets

#./src/dhlp//process_offset_dist.py ./src/dhlp/config/test.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth
#./src/dhlp//process_offset_sing.py ./src/dhlp/config/testCluster.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth

#./src/dhlp//process_offset_sing_hungarian.py ./src/dhlp/config/testCluster.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth

#
#./src/dhlp//process_offset_sing_try.py ./src/dhlp/config/testCluster.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth

./src/dhlp//process_offset_sing_try.py ./src/dhlp/config/test.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best_c.pth



#./src/dhlp//process_offset_dist.py ./src/dhlp/config/testCluster.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth

#
#./src/dhlp/process_offset_sing_mae.py ./src/dhlp/config/test.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth


#./src/dhlp/process_offset_sing_mae1.py  ./src/dhlp/config/test.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth


#Final Average Offset Errors (Nearest Junction Based):
#Avg Mean Offset: 24.49
#Avg Lower Offset: -10.44
#Avg Upper Offset: 59.41


####------------------------
# create the dataset as dhlp format for the crops

#./src/dhlp/dataset/wireframe_all_test.py data/synthetic_plots/multi_cat/testing/m_crops data/dhlp/pcw_alltest


#-----------------noisy inputs of test data (without unet)
# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json

#./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/color data/dhlp/pcw_ntest

# Step 2
# create masks for test data
#./src/dhlp//dataset/gen_mask.py data/dhlp/pcw_ntest/test data/dhlp/pcw_ntest/masks

#%% ------------ noisy cluster peak images ----------
# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json

#./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/color data/dhlp/pcw_ntest

# Step 2
# create masks for test data
#./src/dhlp//dataset/gen_mask.py data/dhlp/pcw_ntest/test data/dhlp/pcw_ntest/masks


#%% ------------ denoised unet cluster  ----------
# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json

./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/cluster data/dhlp/pcw_test_cls

# Step 2
# create masks for test data
./src/dhlp//dataset/gen_mask.py data/dhlp/pcw_test_cls/test data/dhlp/pcw_test_cls/masks
