# SceneTransformer 
Link to the paper: (https://arxiv.org/abs/2106.08417)


## Environment Setup 
 
1. Install conda (with python=3.10)

2. Install pytorch 1.13 cuda version. 

	- Here is an example command of pytorch installation in conda for CUDA 11.7: 

    	`conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia`

3. Check if torch is able to make use of CUDA
	
	- This can be done using `torch.cuda.is_available()`

4. Install PyTorch Lightning 1.8 (supports PyTorch 1.10, 1.11, 1.12 and 1.13) 
    `pip install pytorch-lightning=1.8`

    - Ensure that pytorch lightning version being used is according to the compatibility matrix: https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix

5. Install hydra 
	- follow https://github.com/facebookresearch/hydra/issues/920#issuecomment-702700424

6. In case there is an issue as shown below: 
	![](https://paper-attachments.dropboxusercontent.com/s_744C9F65E126F4E98BAEA6A98A7E8E4EB03ACD47635CC553A65BC24A78795085_1679777039274_Screenshot+2023-03-26+at+2.13.56+AM.png)


	Install this: `pip install "protobuf==3.20.*”`

7. Install cv2: `conda install -c conda-forge opencv`

8. Install tfrecord: `pip install tfrecord`

## Data Preparation

- Download waymo motion data from: https://waymo.com/intl/en_us/open/download/
	- v1.1 data was used for experiments

- The data is partitioned into training, validation, test at the official source using 70/15/15 split. 
	- Partial data used for current experiments: 
		- Train: First 66 tf record files 
		(`training_tfexample.tfrecord-00000-of-01000` to `training_tfexample.tfrecord-00065-of-01000`)
		- Val: First 10 tf record files
		(`validation_tfexample.tfrecord-00000-of-01000` to `validation_tfexample.tfrecord-0009-of-01000`)

- idx files need to be generated for tf.record files using 
	- Use `datautil/create_idx.py` to create a folder of idx files by processing a folder of tf.record files
		- Generate train idx files
		- Generate val idx files

- Visualize a sample of train data using `datautil/Waymo_Open_Dataset_Motion_Tutorial.ipynb`

Resources: 
- Sample motion data: https://waymo.com/intl/en_us/open/data/motion/tfexample
- Summary of motion dataset: https://waymo.com/intl/en_us/open/data/motion/
- TFrecord tutorial: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_motion.ipynb
- Waymo open Dataset paper: https://arxiv.org/abs/2104.10133 


## Project Structure Overview

- `main.py` - Run this in order to train/test/validate the model
- `conf/config.yaml` - Set the configuration of dataset, model and training. Check comments for definitions. 
- **datautil**
	- `create_idx.py` - Use it to create indices from tfrecord data
	- `waymo_dataset.py` - Defines the dataset class, dataloader and the collate function. 
- **model**
	- `encoder.py` - Defines the encoder of SceneTransformer
	- `decoder.py` - Defines the decoder of SceneTransformer
	- `pl_module.py` - Defines the training, validation, testing parts. Also contains visualization code. 
	- `utils.py` - Model/data/training utilities
- **tfrecordutils** - tf record utilities
- **outputs/** - Outputs of train/test are stored here based on the data and time of the run. 

## Train a model 
Training code loads data as per the dataloader and learns through backpropagation of loss calculated between ground truth and predictions obtained by the model. 

- Download the dataset and prepare train, val data and idxs in seperate folders following the procedure shared above.  
- In the config.yaml file, set mode to train. 
	- Set the dataset paths under dataset field of config
	- Set the model's parameters under the model field. 
	- Make sure the gpu's set in config are available for use. 
	- Run `main.py` to start training the model 
- The code automatically does a sample validation initially and then proceeds to train the model. Checkpointing of train (and val) loss, metrics, visualizations are done through tf events file. Use `tensorboard` to visualize the relevant plots and validation images. 
- Training can be resumed from a previous checkpoint by setting the path of checkpoint in resume field of `config.yaml`
	- Depending on pytorch lightning version installed, there might be an error when resuming training. In that case, comment out the `validation_epoch_end` function in `pl_module.py` and resume training. 


## Test a model
Test code processes a folder of tf.record data to produce an output folder containing visualizations of each scene.

- Generate idx files of test data as described above using `datautil/create_idx.py`
- In the config.yaml file, set mode to test
- Under the dataset's test field of config file, do the following: 
	- Set the paths to test tf.record folder and idx folder
	- Set the number of seconds into the future model should predict using `prediction_time`
	- Set the path to model weights using `ckpt_path`
	- Run main.py to start testing the model 
	- The evaluation metrics (minFDE, minADE, inference time etc) of test data are printed at the end and output visualizations (.png files) are saved in outputs folder. 

## Differences in training conditions
The following changes have been made in training conditions when compared to training conditions used in the paper: 

- `feature_dim`: 256 -> 64
- `batch_size`: 64 -> 2
- `in_feat_dim`: 7 -> 9
- Data augmentation: None (Paper uses data augmentation)
- Partial training and validation data (~7% of original data used in the paper, details shared above in Data Preparation)
- MSE Loss is used instead of multiple losses (Displacement, classification, Laplace, heading losses) used in the paper

