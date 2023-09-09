# ADPT-AI

## Overview
For my master's thesis I decided to approach the **automatic detection of persuation techniques on social networks**.   
To tackle the problem we employed machine learning, deep learning and natural language processing techniques. In the end we achieved interesting results although far from perfect.

Our model's final architecture is represented by:
![Final model architecture](images/model-architecture.png)

## Repository structure
The repo contains the data used to train the model in the _data_ folder. 

The _docs_ folder contains both the thesis and the slides for the thesis presentation. 

The _src_ directory contains the code for each iteration of the project.
Some iterations are missing from the repo but we'll try to add them soon. The code is also a bit messy accross some files but we'll try to fix it in the future.

## Future work
To improve the model's capability we must collect more annotated data in order to train the model, that way we can make it more robust and capture relevant patterns.

Currently the model only supports textual elements classification. It would also be interesting to add:
- **Optical character recognition**: by adding this feature the text would be extracted from the image and latter fed to the model.
- **Multimodal classification**: creating a multimodal model that can classify both images and text would make the model much more powerfull. 