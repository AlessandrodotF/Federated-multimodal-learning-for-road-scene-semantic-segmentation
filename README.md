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

In this work we propose a novel realistic scenario for Semantic Segmentation in Federated Learning: Federated
source-Free Domain Adaptation (FFreeDA). In FFreeDA, the server can pre-train the model on labeled source data.
However, as in the Source-Free Domain Adaptation (SFDA) setting, further accessing the source data is forbidden.
Clients can access only their **unlabeled** target dataset, but cannot share it with other clients nor with
the server. Moreover, there are many clients in the system and each of them has only a limited amount of images,
to emulate real-world scenarios. Therefore, after the pre-training phase, the training is fully unsupervised.
To address the FFreeDA problem, we propose LADD, a novel federated algorithm that assumes the presence of 
multiple distributions hidden among the clients. LADD partitions the clients into clusters based on the styles of the
images belonging to each client, trying to match them with their actual latent distribution. LADD shows excellent
performance on all benchmarks with a source dataset (GTA5) and three different targets (Cityscapes, CrossCity,
Mapillary), with diversified splits of the data across the clients.

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



