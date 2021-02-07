# Autoencoder-based File Sorter

## Introduction

This is a PyTorch implementation of an autoencoder-based file sorter. The eventual goal of this is to have autoencoders that can be used to sort and classify both image and audio data, after being trained on a sufficient use case. My intention with it is to sort a large collection of desktop backgrounds (2K+ files) based on similarity of their encodings. If this is is reasonably successful, I will also use it to clean up some messy datasets I have for a GAN project I'm working on.

## Current State

Currently, basic dataset utilities and a convolutional autoencoder for images have been implemented.

## Todo

(In no particular order)

* Add audio autoencoder
	* add utils for loading audio files
	* devise methods for working with audio
		* 1D convolutions?
		* arrays of FFTs?
* Data point embedding functionality
	* Method to go from encoded image/audio to an N-D embedding space
	* Methods to find proximity in space
	* Methods to visualize points in space
		* PCA?
		* t-SNE?
* Clean up code
	* Remove hardcoding
	* Ensure good commenting
* General utilities
	* add optionality for more:
		* loss functions
		* optimizers
		* transformations (e.g. image augmentation)
	* functionality to switch dataset mode (i.e. image to audio and vice-versa)
* Dashboard
	* [Probably based on this](https://dash-gallery.plotly.host/dash-tsne/)
	* Ideally 2/3D chart with files represented as points
	* Proximity based on image distance
	* Clicking image displays it
	* Method to enter new image path, run through encoder, and see its embedding
* Advanced Add-Ons
	* Distillation on [CLIP](https://github.com/openai/CLIP) in image training to facilitate image "understanding"
