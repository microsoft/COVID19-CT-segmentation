# Effective Deep Learning Approaches for Predicting COVID-19 Outcomes from Chest Computed Tomography Volumes
This project provides code to train semantic segmentation models for pulmonary lesions segmentation from Computer Tomography (CT) scans.

# Getting Started
## Data preprocessing and Dataset creation
1. **Mask Generation**: Generate the segmentation mask for the different classes.
2. **Random Split**: The final dataset is saved in three folders: ``train/ test/ val/``
3. **Generate stats**: Generate statistics of the input image channels: returns keys for means and stds accross the channels in input slices. Note: Not implemented yet.
<br />

## Software Dependencies
🐍 Python 3.x
* torch==1.4.0
* torchvision==0.5.0
* tifffile==2020.2.16
* opencv-python==4.2.0.32
* matplotlib==3.1.3
* pandas==1.0.1
* pyarrow==0.16.0
* imutils==0.5.3
* scikit-image==0.16.2
* scikit-learn==0.22.2
* tqdm==4.46.1
* xmltodict==0.12.0

## Install Dependencies with Conda 

```bash
conda create --name ct_segmentation python=3.6
conda activate ct_segmentation
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

Note: Make sure to intall a version of cudatoolkit supported by your NVCC version

# Build and Test
To run all preprocessing and create dataset:

``python3 -m data.preprocess_dataset.py --raw_data_dir=/path_to_the_directory_where_raw_data_is_located_at --output_dir=/path_to_the_directory_where_covid_ct_datadset_will_be_located_at/``

This will generate a dataset in the format:
```
output_dir/covid_ct/
└── train/
    └── images/
    └── masks
└── val/
    └── images/
    └── masks/
└── test/
    └── images/
    └── masks/
```

### Training multihead segmentation Network

To train the network with default values and see training output on the command line run:

``python3 train.py --data_dir=/dir_to_datasets_directory/ --verbose`` 

Note: Make sure the path to your code is added to th PYTHONPATH, so modules can be recognized: ``export PYTHONPATH=$PYTHONPATH:/path_to_project_code_folder/``

# Contribute
You can contribute to the project by submitting bugs and feature requests. 


# License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT license.