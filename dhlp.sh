# Step 1
# generate the npz for each image using the category separated images and the lines json present in train and valid json

./src/dhlp/dataset/wireframe_noisy.py data/synthetic_plots/multi_cat/training/color data/dhlp/pcw_1n

# Step 2
# once the data is present provide the same path in the yaml where data is preset and start the training

#python ./src/dhlp/train.py --identifier baseline ./src/dhlp/config/noisy.yaml