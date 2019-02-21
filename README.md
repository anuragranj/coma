# CoMA: Convolutional Mesh Autoencoders

![Generating 3D Faces using Convolutional Mesh Autoencoders](http://coma.is.tue.mpg.de/assets/coma_faces.jpg)

This is an official repository of [Generating 3D Faces using Convolutional Mesh Autoencoders](https://coma.is.tue.mpg.de)

[[Project Page](https://coma.is.tue.mpg.de)][[Arxiv](https://arxiv.org/abs/1807.10267)]

## Requirements
This code is tested on Tensorflow 1.3. Requirements (including tensorflow) can be installed using:
```bash
pip install -r requirements.txt
```
Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh).

## Data
Download the data from the [Project Page](https://coma.is.tue.mpg.de).

Preprocess the data
```bash
python processData.py --data <PATH_OF_RAW_DATA> --save_path <PATH_TO_SAVE_PROCESSED DATA>
```

Data pre-processing creates numpy files for the interpolation experiment and extrapolation experiment (Section X of the paper).
This creates 13 different train and test files.
`sliced_[train|test]` is for the interpolation experiment.
`<EXPRESSION>_[train|test]` are for cross validation cross 12 different expression sequences.

## Training
To train, specify a name, and choose a particular train test split. For example,
```bash
python main.py --data data/sliced --name sliced
```  

## Testing
To test, specify a name, and data. For example,
```bash
python main.py --data data/sliced --name sliced --mode test

```
#### Reproducing results in the paper
Run the following script. The models are slightly better (~1% on average) than ones reported in the paper.

```bash
sh generateErrors.sh
```

## Sampling
To sample faces from the latent space, specify a model and data. For example,
```bash
python main.py --data data/sliced --name sliced --mode latent
```
A face template pops up. You can then use the keys `qwertyui` to sample faces by moving forward in each of the 8 latent dimensions. Use `asdfghjk` to move backward in the latent space.

For more flexible usage, refer to [lib/visualize_latent_space.py](https://github.com/anuragranj/coma/blob/master/lib/visualize_latent_space.py).

## Acknowledgements
We thank [Raffi Enficiaud](https://www.is.mpg.de/person/renficiaud) and [Ahmed Osman](https://ps.is.tuebingen.mpg.de/person/aosman) for pushing the release of [psbody.mesh](https://github.com/MPI-IS/mesh), an essential dependency for this project.

## License
The code contained in this repository is under MIT License and is free for commercial and non-commercial purposes. The dependencies, in particular, [MPI-IS/mesh](https://github.com/MPI-IS/mesh) and our [data](https://coma.is.tue.mpg.de) have their own license terms which can be found on their respective webpages. The dependencies and data are NOT covered by MIT License associated with this repository.

## When using this code, please cite

Ranjan, Anurag, Timo Bolkart, Soubhik Sanyal, and Michael J. Black. "Generating 3D faces using Convolutional Mesh Autoencoders." ECCV 2018.
