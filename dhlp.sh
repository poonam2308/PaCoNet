# step 1
# generate np for each image using crops_white images and the lines present in train and valid json
./src/dhlp/dataset/wireframe_crops.py data/synthetic_plots/multi_cat/training/crops_white data/dhlp/pcw_crops

# Step 2
# once the data is present provide the same path in the yaml where data is preset and start the training
#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/crops.yaml


# Step 1
# generate the npz for each image using the category separated images and the lines json present in train and valid json
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

#./src/dhlp/dataset/wireframe_denoised.py data/synthetic_plots/multi_cat/training/color data/dhlp/pcw_2dn


# step 2
#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/denoised.yaml


#%-----------------------------color-----------------------------%
# this is for the denoised dataset which are generated with the right set of line coordinates - mapped to the lines for color based separation
##step 1
#./src/dhlp/dataset/wireframe_denoised.py data/synthetic_plots/multi_cat/training/color data/dhlp/pcw_color

# step 2
#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/denoised.yaml

#%-----------------------------cluster-----------------------------%

# this is for the denoised dataset which are generated with the right set of line coordinates - mapped to the lines  for clustererd based separation
#step 1
#./src/dhlp/dataset/wireframe_denoised.py data/synthetic_plots/multi_cat/training/cluster data/dhlp/pcw_cluster

# step 2
#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/denoisedCluster.yaml


# create dhlp format dataset for the ground truth data
# step 1
#./src/dhlp/dataset/wireframe_gt.py data/synthetic_plots/multi_cat/training/gt/gt_cat_rename  data/dhlp/pcw_gt

# step 2
#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/gt.yaml

# create dhlp format dataset for without unet color peak data
# step 1
#./src/dhlp/dataset/wireframe_noised.py data/synthetic_plots/multi_cat/training/color  data/dhlp/pcw_ncolor

# step 2
#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/noisedPeak.yaml


# create dhlp format dataset for without unet dbscan peak data

# step 1
#./src/dhlp/dataset/wireframe_noised.py data/synthetic_plots/multi_cat/training/cluster  data/dhlp/pcw_ncluster

# step 2
#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/noisedCluster.yaml




