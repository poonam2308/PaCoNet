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
# calculate sap, map and mean offsets

#./src/dhlp/eval-sAP.py ./outputs/results_test1

#./src/dhlp/eval-mAP.py ./outputs/results_test1
#Evaluating ./outputs/results_test
#  27.4 | 38.4
#Evaluating ./outputs/results_test1
#  30.9 | 44.0

# to get the mean error
#
#./src/dhlp//process_offset_dist.py ./src/dhlp/config/test.yaml ./outputs/logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth

####------------------------
# create the dataset as dhlp format for the crops

./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/m_crops data/dhlp/pcw_alltest



