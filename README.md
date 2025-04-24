# Federated multimodal learning for road scene semantic segmentation
This work is an extension of *Learning Across Domains and Devices: Style-Driven Source-Free Domain Adaptation in Clustered Federated Learning*
(**Official implementation** of [Learning Across Domains and Devices: Style-Driven Source-Free Domain Adaptation in
Clustered Federated Learning](https://arxiv.org/abs/2210.02326) by **Donald Shenaj**<sup>\*,1</sup>, **Eros Fanì**<sup>\*,2</sup>, Marco Toldo<sup>1</sup>,
Debora Caldarola<sup>2</sup>, Antonio Tavera<sup>2</sup>, Umberto Micheli<sup>&#8224;,1</sup>,
Marco Ciccone<sup>&#8224;,2</sup>, Pietro Zanuttigh<sup>&#8224;,1</sup> and Barbara Caputo<sup>&#8224;,2</sup>
**Corresponding authors:** donald.shenaj@dei.unipd.it, eros.fani@polito.it

<sup>\*</sup> Equal contribution. <sup>&#8224;</sup> Equal supervision.
<sup>1</sup> Authors supported by University of Padova, Padua, Italy.
<sup>2</sup> Authors supported by Politecnico di Torino, Turin, Italy.)



## Summary
Federated learning (FL) is a machine learning approach that allows multiple devices, also referred to as clients, to collaboratively train a shared model without exchanging their data or
any kind of sensitive information. In this framework, instead of transmitting data to a central
server for the training procedure, the global model is trained locally on each device and only
the updates are shared. These updates are aggregated according to ad-hoc strategies to improve
the shared model performances on the server side.
This project is built upon such framework and its primary scope is to explore potential applications in the field of semantic segmentation for road scenes. The starting point is represented
by the results and the implementation made in *Learning Across Domains and Devices: Style-Driven Source-Free Domain Adaptation in Clustered Federated Learning* , on this way a new scenario is proposed:
addressing the semantic segmentation task in a multimodal setting. The main idea is that different data representations for the same road scene can be used to enhance the ability of the
algorithm to learn representations and correctly classify the pixels in the given image.
In thiswork awell-known dataset for semantic urban scene understanding, namely Cityscapes, is used in its original RGB format. Beside this, a second version of the dataset, containing
additional geometrical information, is generated with a simple python script and it used in the
different experiments. The semantic segmentation task is tackled using an encoder-decoder architecture: in all the experiments there will be one (or more) encoder implemented in terms of
Mobilenet-v2 and Deeplabv3 for the decoder part. The former is used to compress the input
image in a low dimensional space trying to extract the most relevant features of the input, it is
designed to run on mobile devices or, in general, on low power hardware as the case of interest.
The latter is used to classify the output of the encoder trying to predict the correct class for each
pixel in a supervised setting.


## Setup

### Preliminary operations

1) Clone this repository.
2) Move to the root path of your local copy of the repository.
3) Create the ```LADD``` new conda virtual environment and activate it:
```
conda env create -f environment.yml
conda activate LADD
```

### Datasets

1) Download the the Cityscapes dataset from [here](https://www.cityscapes-dataset.com/downloads/) (```gtFine_trainvaltest.zip``` and ```leftImg8bit_trainvaltest.zip``` files).
2) Ask for the Crosscity dataset [here](https://yihsinchen.github.io/segmentation_adaptation_dataset/).
3) Download the Mapillary Vistas dataset from [here](https://www.mapillary.com/dataset/vistas).
4) Download the GTA5 dataset from [here](https://download.visinf.tu-darmstadt.de/data/from_games/).
5) Extract all the datasets' archives. Then, move the datasets' folders in ```data/[dataset_name]/data```, 
where ```[dataset_name]``` is one of ```{cityscapes, crosscity, mapillary, gta5}```.
```
data
├── cityscapes
│   ├── data
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── splits
├── gta5
│   ├── data
│   │   ├── images
│   │   ├── labels
│   ├── splits
├── crosscity
│   ├── data
│   │   ├── cities
│   ├── splits
├── mapillary
│   ├── data
│   │   ├── training
│   │   ├── testing
│   │   ├── validation
│   ├── splits
```

### Experiments' logging

Make a new [wandb](https://wandb.ai/site) account if you do not have one yet, and create a new wandb project.

### How to run

In the configs folder, it is possible to find examples of config files for some of the experiments to replicate the
results of the paper. Run one of the exemplar configs or a custom one from the root path of your local copy of the
repository:

```./run.sh [path/to/config]```

We provide config files to replicate the experiments in the paper:

1) run a pre-training script — e.g. ```./run.sh configs/pretrain_crosscity.txt``` and take the corresponding exp_id from
wandb
2) run our method — e.g. ```./run.sh configs/crosscity_LADD_classifier.txt```. Make sure to set
```load_FDA_id=[exp_id_pretrain]```

N.B. change the ```wandb_entity``` argument with the entity name of your wandb project.

N.B. always leave a blank new line at the end of the config. Otherwise, your last argument will be ignored.



