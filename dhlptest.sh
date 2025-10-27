# Step 1
# output :generate the npz for each image
# input : using the category separated images and the lines json present in  test.json
#
#./src/dhlp/dataset/wireframe_noisy.py data/synthetic_plots/multi_cat/training/color data/dhlp/pcw_1n

# Step 2
# once the data is present provide the same path in the yaml where data is preset and start the training

#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/noisy.yaml


#./process_original.py config/clust5kdenoisednew.yaml logs_clst5kdenew/250224-133604-baseline/checkpoint_best.pth




# this is for the denoised images
# follow step 1 to create the data
#  - use denoised.yaml and wireframe_denoised.py module
# verify the paths are correct
# step 1

./src/dhlp/dataset/wireframe_denoised.py data/synthetic_plots/multi_cat/training/color/denoised data/dhlp/pcw_2dn


# step 2
#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/denoised.yaml