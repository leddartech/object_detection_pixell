# object_detection_pixell

This project contains the model presented with the Pixset dataset paper [LINK]. It performs 3D object detection from point clouds provided by the Leddar Pixell sensor. 

## Requirements
- pioneer.das.api == 1.0.0
- pioneer.common == 1.0.0
- pioneer.das.view == 1.0.0 (optional)
- torch
- ignite
- numpy
- matplotlib
- transforms3d
- yaml
- numba

## Dataset preparation

The Pixset dataset can be downloaded from [LINK]. Next, unzip all sequences. The test set consists of the sequences that contain "part*" in their name, where * is 1, 9, 26, 32, 38 and 39. Put the sequences for the test set in a directory, then all remaining sequences in a separate directory for the train set.

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