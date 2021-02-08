# Autoencoder-based File Sorter

## Introduction

This is a PyTorch implementation of an autoencoder-based file sorter. The eventual goal of this is to have autoencoders that can be used to sort and classify both image and audio data, after being trained on a sufficient use case. My intention with it is to sort a large collection of desktop backgrounds (2K+ files) based on similarity of their encodings. If this is is reasonably successful, I will also use it to clean up some messy datasets I have for a GAN project I'm working on.

## Current State

Currently, basic dataset utilities and a convolutional autoencoder for images have been implemented, as well as the components for embedding autoencoder encodings and displaying them in 3D space in an interactive dashboard. To run, set a directory folder (filepath variable) in the options/opts.yaml file and then run the app.py file. I will add a requirements.txt file later, currently main requirements are PyTorch, pandas, plotly, dash, and SKLearn (as well as any of their requirements).

## Acknowledgements

In general the many writers for Python, pandas, PyTorch, SKLearn, plotly, and Dash. In particular, I based my dashboard heavily on [this one](https://dash-gallery.plotly.host/dash-tsne/), with reductions in content and changes to image asset loading.

## Todo

(In no particular order)
* Add more visual autoencoder features
	* Architecture options
		* Resnet/skip connections
		* More layer options
* Add audio autoencoder
	* add utils for loading audio files
	* devise methods for working with audio
		* 1D convolutions?
		* arrays of FFTs?
* Data point embedding functionality
	* Additional methods for embedding
		* PCA?
		* t-SNE?
* General utilities
	* Add requirements.txt (probably waiting until all the basics are in before this)
	* add optionality for more:
		* loss functions
		* optimizers
		* transformations (e.g. image augmentation)
	* functionality to switch dataset mode (i.e. image to audio and vice-versa)
* Dashboard
	* Clean up, in particular change layout for better aesthetics
	* Fully solve issue with html decoding
	* Method to enter new image path, run through encoder, and see its embedding
* Advanced Add-Ons
	* Distillation on [CLIP](https://github.com/openai/CLIP) in image training to facilitate image "understanding"
