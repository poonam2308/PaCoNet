#---------------------------------- color------------------------------------------------------------------------------
# Step 1 :generate the predictions in npz and the path where they are generated are used in the run command for eval-sap

# Step 2  :use the path where the pred are generated using step 1
#./src/dhlp/eval-sAP_original.py ./outputs/results/c

#---------------------------------- cluster-----------------------------------------------------------------------------
#./src/dhlp/eval-sAP_original.py ./outputs/results/cls

#----------------------------------- color no unet-----------------------------------------------------------------------

#./src/dhlp/eval-sAP_original.py ./outputs/results/nc

#----------------------------------- cluster no unet----------------------------------------------------------------------

#./src/dhlp/eval-sAP_original.py ./outputs/results/ncls
#-------------------------------------------------------------------------------------------------------------------------