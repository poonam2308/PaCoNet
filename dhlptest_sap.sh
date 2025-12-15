#---------------------------------- color------------------------------------------------------------------------------
# Step 1 :generate the predictions in npz and the path where they are generated are used in the run command for eval-sap
#./src/dhlp/process_original.py ./src/dhlp/config/test.yaml ./outputs/chkpt/dhlp/checkpoint_best_c.pth

# Step 2  :use the path where the pred are generated using step 1
./src/dhlp/eval-sAP.py ./outputs/results/c

#----------------------------------- cluster---------------------------------------------------------------------------
#./src/dhlp/process_original.py ./src/dhlp/config/testCluster.yaml ./outputs/chkpt/dhlp/checkpoint_best_cls.pth

#./src/dhlp/eval-sAP.py ./outputs/results/cls

#----------------------------------- color no unet-----------------------------------------------------------------------
#./src/dhlp/process_original.py ./src/dhlp/config/noisedTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_nc.pth
##
#./src/dhlp/eval-sAP.py ./outputs/results/nc

#----------------------------------- cluster no unet----------------------------------------------------------------------
#./src/dhlp/process_original.py ./src/dhlp/config/noisedClusterTest.yaml ./outputs/chkpt/dhlp/checkpoint_best_ncls.pth

#./src/dhlp/eval-sAP.py ./outputs/results/ncls
#-------------------------------------------------------------------------------------------------------------------------