### Deep Hough Transform using Line priors
    - The category separated images from parallel coordinates based on the color histogram, all the line coordinates are saved in one all_data.json and split into train and valid.json 

### config folder
    - configuration yaml file contains the path and model details 

### evaluation 
    - line matching 
        - msTPFP() matches predicted lines to ground truth lines based on the distance between midpoints 
        - for each predicted line - if its midpoint is within threshold pixles of a ground truth midpoint and that ground truth has not been matched yet - it is a true positive 
        - otherwise it is a false positive (FP)
        - to prevent multiple predictions from being counted for the same ground truth
    - metric computation 
        - compute_ap
        - tp : an array where each entry is 1 if that prediction is a true positive else 0 e.g. [1,0,1,0]
        - fp : an array where each entry is 1 if that prediction is a false positive else 0 e.g. [0,1,0,1]
        - Cumulative precision and recall 

    

