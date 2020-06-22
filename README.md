# Analysis of Neural Networks. Applications to Interpretability and Uncertainty | Code
<br/>

This repository contains the code implemented for the Final Project ```Analysis of Neural Networks. Applications to Interpretability and Uncertainty```. The code has been written in ```Python``` and is divided in different ```Jupyter Notebooks```.

### Requirements
This repository contains a file named ```environment.yml``` with the ```tfg``` environment, which includes all required packages:
```bash
conda env create -f environment.yml
conda activate tfg
```
The library used to implement the neural networks is [PyTorch](https://github.com/pytorch/pytorch).
The library used to implement the interpretability algorithms is [captum](https://github.com/pytorch/captum).
### Notebooks
Here is the list of notebooks containing the code:
- ```classifier_circles.ipynb```: Classification problem of two concentric circles.
- ```classifier_segment.ipynb```: Classification problem of m subsets in a 1d segment.
- ```ae_latent_space_mm.ipynb```: Autoencoder for stock market time series under a normalisation transformation. Also latent space explorer.
- ```ae_latent_space_st.ipynb```: Autoencoder for stock market time series under a standardisation transformation. Also latent space explorer.
- ```ae_results.ipynb```: Results analysis for the models trained in the two previous notebooks.
- ```interpretability.ipynb```: Implementation of Integrated Gradients, Occlusion and CAM and test in image.
- ```uncertainty.ipynb```: Two methods for adding uncertainty to a regression problem.

### Images
The images folder contains the original plots and gifs from the project.
