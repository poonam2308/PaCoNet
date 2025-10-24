### Parallel coordinate data generation
    - Raw data generated using  real wrold distribution (one time process) in  sythetics csvs
    - Using csvs, synthetic plots are created with meta data information (one time process).
### Meta data syntax:
```
filename": "image_1.png",
      "vertical_axes": [
          105.0,
          ...
      ],
      "category_colors": {
          "SOXR": {
              "h": 0.6,
              "s": 1,
              "v": 1
          },
          "y1hE7": {
              ...
          }
      },
      "lines": {
          "crop_1": [
              [
                  0.0,
                  221.54,
                  200.0,
                  153.46
              ],
              ...
          ],
          "crop_2": [
              ..
          ] 
     }
```

### Cropping
    Once the plots are created the directory where the data is present used for cropping.
    - cropping is performed by reading images from the main image directory 
    - cropped images are cropped using vertical axes from the meta data json file
    - cropped images are saved in cropping_dir

### Category extraction 
    Once the crops are present in a cropped images direcotry
    - each crop is read from cropping_dir and and meta data json file based on image index is looked into for the lines in the json dir 
    - path of json should be provided along with the images 
    - it is possible that users might not have the color information in the meta data json 
    - when there is no color information then, categories from each crop should be extracted based on color historgam peaks
    - And once you know the peaks ,separate the categories and sved them individually in a new image 
    - the new image must be of the same size as original crop and give a cat number based on how many peaks are present
    - all the categories separated images are saved in a new category_dir
    - once all the categories are separated then we need to find the all the lines coordinates for the categories 
    - in the meta data json the lines coordinates are the list per crop 
    - we need to find which lines are for which cat
    - this can be perfomed using the masking and mapping the pixel for that particulary categories 
    - now that we have lines coordinates mapped per categories, they must be saved in a new json which contains all the data 
    

    #### doubts for category separation:
        - where do we save the lines json for each category 
        - do we save in new json per image ?
        - do we save in one compiled json for all images ?
        - do we update the existing meta data ( very bad option, i think) 

### Denoising 
    once the categories are present in a category separated images directory category_dir
    - either use them directly for the DHLP 
    - or denoise them and save them in a new folder named denoised_category_dir
    - the denoised images maybe different in size based on unet, scale the line coodrinates accordingly 
    - lines coordiantes in any case are requried by DHLP for training


    



