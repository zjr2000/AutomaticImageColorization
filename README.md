# AutomaticImageColorization
 Automatic Image Colorization。SUSTech CS308 final project

 ## Get Started
 ```
 pytorch==1.7.1
 pip install einops
 pip install scikit-image
 ```
## Requirement data
 ```
Download the data from: http://places2.csail.mit.edu/download.html
mkdir ./data | mkdir ./logs | mkdir results
mkdir ./results/checkpoints | mkdir ./results/pred
mv download_dataset ./data/
 ```
After downloading the data, put it in './data/places10', or modify the path in data_loader.py.

## Train

After configuring the environment and downloading the dataset, you can start training the model.
Detailed configuration parameters can be found in config.yaml.
 ```
bash train.sh
 ```
 
## Test
The accuracy of the model was tested by L1, L2, raw, ACC and other methods.
```
bash eval.sh
 ```
## Colorization
Colorizate images in './data/places10/test' path
 ```
bash colorization.sh
 ```
## Exchange style
You can only exchange the style of one picture at a time. 
Change the style of the first picture to the second.
You need to unlocks the code at line 56 in colorize.py.
 ```
bash colorization.sh
 ```
