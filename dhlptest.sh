# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json

#./src/dhlp/dataset/wireframe_test.py data/synthetic_plots/multi_cat/testing/color data/dhlp/pcw_test



# Step 2
# create masks for test data

./src/dhlp//dataset/gen_mask.py data/dhlp/pcw_test/test data/dhlp/pcw_test/masks



# Step 3
#generate the predictions using the best checkpoint for the test data

#./src/dhlp/process_original.py ./src/dhlp/config/test.yaml logs_ct5k1/250223-200052-baseline/checkpoint_best.pth



