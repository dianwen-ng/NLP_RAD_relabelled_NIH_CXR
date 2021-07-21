## Getting Started  

Step 1. Clone or download this repository and set it as the working directory, create a conda environment and install the dependencies.

```
cd NLP_RAD_relabelled_NIH_CXR
conda create -n nlpvsrad python=3.6
conda activate nlpvsrad
pip install -r requirements.txt 
```

## Make Data Manifest for Train/Test Split (Re-labelled from Radiologist)
Step 2. The dataloader from this system requires a JSON manifest to process. To create the data files, you can run the following command in your command line.
```
python data/make_data.py <root directory to NIH image>
e.g. python data/make_data.py /data/volume03/NIH-chest/NIH
```

## Training
Step 3. To train the model, use main.py with specific model selections. \n
Suggested training learning rates are
- EfficientNet: 1e-5
- DenseNet: 1e-5
- ResNet: 1e-7

```
e.g.
python main.py -model efficientnet -save_dir save_models/efficientnet 

```