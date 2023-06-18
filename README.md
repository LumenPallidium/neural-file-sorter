# Neural File Sorter

## Introduction

This is a PyTorch implementation of a file sorter that uses neural networks to sort images based on perceptual similarity. The eventual goal of this is to have networks that can be used to sort and classify both image and audio data, after being trained on a sufficient use case. My uses for it have been sorting wallpapers on my computer and cleaning datasets used for GANs.

## Installation

First, follow the instructions for installing [OpenAI's CLIP model](https://github.com/openai/CLIP), if you are going to use it. Note that I am using torch 2.0, though using an early version down to 1.7.1 shouldn't cause problems. You will need torch regardless of whether you install CLIP.

The other libraries required are in the requirements.txt file, so install as you prefer. torch is in the requirements file, but I would reccomend installing independently since it can be very hardware dependent. Briefly, the other major libraries used are:

* scikit-learn
* pandas
* plotly
* dash

As usual, the requirements file can be installed by navigating to this folder and running:

```
pip install -r requirements.txt
```

## Features

Currently, basic dataset utilities, a convolutional autoencoder and [OpenAI's CLIP model](https://github.com/openai/CLIP) for image encoding have been implemented, as well as the components for embedding network encodings and displaying them in 3D space in an interactive dashboard.  To run, set a directory folder (filepath variable) in the options/opts.yaml file and then run the app.py file.

As some use cases for this repo:

- Grouping photos based on their content (see app.py, embedding.py, embedding_clip.py)
	- This can be done using CLIP or a custom-trained autoencoder
	- CLIP is much more powerful and doesn't need to be trained, so it is reccomended
	- This can be done with manually-coded labels (using clip_categories in the config) or in an unsupervised manner
- Organizing photos on your drives (see reorganize.py)
	- Given the categorizations/classifications created by embedding.py, you can automatically sort images into folders
	- Hence, the title of this repo
	- I've used this for sorting images for use with a GAN, which avoids manual curation of thousands of images

## Example Video

https://user-images.githubusercontent.com/42820488/116844734-cf865c80-ab98-11eb-8caa-71abcb9430ae.mp4

The video above shows the Dashboard for interacting with your file embeddings. This example is of desktop backgrounds on my computer. I start by selecting images classified as "forest", by the classifier. Notice that "forest" and "city" are at opposite ends of the embedding axis, as one might expect. Also notice how powerful CLIP can be with user-defined categories: at 0:49 in the video, I begin going through images labelled as "comfy". In line with the intuitive noton of "comfiness", this includes pictures of soft fluffy animals, conversations on a rooftop, and rooms that are cluttered but welcoming.

## Acknowledgements

In general, OpenAI and the many writers for Python, pandas, PyTorch, SKLearn, plotly, and Dash. In particular, I based my dashboard heavily on [this one](https://dash-gallery.plotly.host/dash-tsne/), with reductions in content and updates to image asset loading.



