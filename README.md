# object_detection_pixell

This project contains the model presented with the Pixset dataset paper [LINK]. It performs 3D object detection from point clouds provided by the Leddar Pixell sensor. 

## Installation
After cloning this repo, run
```
cd object_detection_pixell
python3 setup.py develop --user
```

## Inference
Live inference is possible with the provided Predictor class. Here is a usage example:
```
from object_detection_pixell.predictor import Predictor

config = 'object_detection_pixell/configs/pixell_to_box3d_v2.yml'
state = # Put the path to a .pt file containing the weights of the model

predictor = Predictor(config, state)

point_cloud = # Get a point cloud from a Pixell sensor as a Nx3 numpy array.
amplitudes = # Get the corresponding amplitudes of the point cloud as a Nx1 numpy array.

predictions = predictor(point_cloud, amplitudes)
```

Pre-trained weights (.pt file) can be provided (send request to jean-luc.deziel@leddartech.com).

## Dataset preparation (for training and testing)

The Pixset dataset can be downloaded from \url{dataset.leddartech.com}. Next, unzip all sequences. The test set consists of the sequences that contain "part*" in their name, where * is 1, 9, 26, 32, 38 and 39. Put the sequences for the test set in a directory, then all remaining sequences in a separate directory for the train set. Add the paths to the train and test set to the indicated lines in configs/pixell_to_box3d.yml.

Then, run 
```
python3 prepare_inserts_data.py --cfg=configs/pixell_to_box3d.py
```
This will generate the data necessary for data augmentattion.

## Training

Run 
```
python3 train.py --cfg=configs/pixell_to_box3d.py
```
If at some point, you encounter errors mentionning "BadZipFile", you will probably have to unzip the raw data in each sequences. If this happens, please create an issue.

If something goes wrong during the training, you can resume it from any state with
```
python3 train.py --cfg=configs/pixell_to_box3d.py --resume_state=results/*/states/pixell_ech_to_box3d_*.pt
```
Where the * will vary.

## Testing

Run 
```
python3 test.py --cfg=configs/pixell_to_box3d.py --state=results/***/states/pixell_ech_to_box3d_***.pt
```
Where the * will vary. The last * corresponds to the epoch number.

If pioneer.das.view is installed, you can also try to visualize the results with
```
python3 inference.py --cfg=configs/pixell_to_box3d.py --state=results/***/states/pixell_ech_to_box3d_***.pt --dataset=*
```
Indicate the path to the dataset to visualize with the last argument.