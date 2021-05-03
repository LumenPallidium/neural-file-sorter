# Neural File Sorter

## Introduction

This is a PyTorch implementation of a file sorter that uses neural networks to sort images based on perceptual similarity. The eventual goal of this is to have networks that can be used to sort and classify both image and audio data, after being trained on a sufficient use case. My intention with it is to sort a large collection of desktop backgrounds (2K+ files) based on similarity of their encodings. If this is is reasonably successful, I will also use it to clean up some messy datasets I have for a GAN project I'm working on.

## Current State

Currently, basic dataset utilities, a convolutional autoencoder and [OpenAI's CLIP model](https://github.com/openai/CLIP) for image encoding have been implemented, as well as the components for embedding network encodings and displaying them in 3D space in an interactive dashboard. I would reccomend following the instructions on [CLIP's github page](https://github.com/openai/CLIP) for its installation details. To run, set a directory folder (filepath variable) in the options/opts.yaml file and then run the app.py file. I will add a requirements.txt file later, currently main requirements are CLIP, PyTorch, pandas, plotly, dash, and SKLearn (as well as any of their requirements).

## Example Video

https://user-images.githubusercontent.com/42820488/116844734-cf865c80-ab98-11eb-8caa-71abcb9430ae.mp4

The video above shows the Dashboard for interacting with your file embeddings. This example is of desktop backgrounds on my computer. I start by selecting images classified as "forest", by the classifier. Notice that "forest" and "city" are at opposite ends of the embedding axis, as one might expect. Also notice how powerful CLIP can be with user-defined categories: at 0:49 in the video, I begin going through images labelled as "comfy". In line with the intuitive noton of "comfiness", this includes pictures of soft fluffy animals, conversations on a rooftop, and rooms that are cluttered but welcoming.

## Acknowledgements

In general, OpenAI and the many writers for Python, pandas, PyTorch, SKLearn, plotly, and Dash. In particular, I based my dashboard heavily on [this one](https://dash-gallery.plotly.host/dash-tsne/), with reductions in content and updates
 to image asset loading.



