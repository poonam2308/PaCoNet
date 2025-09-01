# Step 1 generate the npz for each image using the category separated images and the lines json present in train and valid json
./src/dhlp/dataset/wireframe_noisy.py data/synthetic_plots/multi_cat/training/color data/dhlp/pcw_1n



#python train.py --identifier baseline config/clust5kdenoisednew.yaml