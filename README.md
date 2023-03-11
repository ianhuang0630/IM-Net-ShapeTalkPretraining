#IM-Net-ShapeTalkPretraining

This page is a duplicate of the [original IMNet pytorch implementation](https://github.com/czq142857/IM-NET-pytorch),
with some additional functionality that facilitates the retraining and data preprocessing for the ShapeTalk dataset.

It also supports the serialization of trained models so that they can be used for downstream inference tasks. This generally
should enable you to use your own virtual environments during inference, and our virtual environment during training.

## Setup
Make a conda environment according to the `environment.yaml` provided.

```
conda env create -f environment.yaml
conda activate imnet
```


## Preprocessing of data

The first thing that needs to be done is to create voxel representations of the occupancy fields of
shapes given the meshes. The way to do that is by running the implicit extraction code in the `shape_representations` repo:
[jupyter notebook](https://github.com/optas/shape_representations/blob/master/shape_representations/notebooks/extract_shape_implicits_for_shapetalk_classes.ipynb)

Make sure to change the values of `top_shapenet_dir`, `top_partnet_dir`, `top_modelnet_dir`, `top_output_dir` to your setting.

This will create a intermediate directory `top_output_dir` holding `.mat`s that contain the voxelizations.

To create the actual inputs into IM-NET, run:

```
python preprocessing_subsample_voxels.py [top_output_dir] [voxel_output_dir] [scaling_pickle_path]
```
where:
- `top_output_dir`: the path to directory with the intermediary outputs from running implicit extraction.
- `voxel_output_dir`: the path to the directory where you would like the post-processed inputs to be, to be used later to retrain IM-NET
- `scaling_pickle_path`: path to the pickle storing individual scaling parameters to each shape sample to better align with the ShapeTalk dataset.



## Setup instructions for using pretrained IM-Net

First, download our pretrained IM-NET weights:

```
cd ckpt_ShapeTalkClasses_pub/
wget http://download.cs.stanford.edu/orion/changeit3d/ckpt_ShapeTalkClasses_pub.zip .
unzip ckpt_ShapeTalkClasses_pub.zip
```

Run `python latents_interface.py` to create a pickle object that can be used to extract and decode IM-NET latents.
Make sure to double check that the default commandline environments within `latent_interface.py` are suitable 
(e.g. the default `data_dir` should be given the `voxel_output_dir` from above)


## Retraining IM-NET:

Retraining requires you to run `bash train_ae_custom.sh`. However, before you do that, make sure that you've edited `train_ae_custom.sh` such that
1. `--data_dir` commandline argument is given thecorrect path to the voxel output dir (see `voxel_output_dir`above)
2. `--splits` commandline argument is given the correct path to the splits csv.
3. `-- checkpoint` commandline argument is given the path to the desired output checkpoint directory for your model weights.

