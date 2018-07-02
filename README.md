# DeepFake_tf
This deepfake using tensorflow is based on the https://github.com/dfaker/df. And it requires the dlib and face-alignment that you can get from https://github.com/1adrianb/face-alignment.git before training.

Inputs are 64x64 images outputs are a pair of 128x128 images one RGB with the reconstructed face, one B/W to be used as a mask to guide what sections of the image are to be replaced.

The same as dfaker, for the reconstrcuted face masked DSSIM loss is used that behaves as a standard SSIM difference measure in the central face area and always returns zero loss in the surrounding background area outside of the face so as as not to train irrelevant features.The project wrote the Dssim code for any frameworked.And MSE is used for the mask-loss which you can see in the model_tf.loss file.
### Guide

* Clone this repository
* make sure that you have satisfied the requirements that listed in the requirements.txt
* make sure that the module of face-alignment can be imoorted.
* run align_images_masked on your source A and B image folders.
* copy the aligned cropped images into the A or B folder along with the alignments.json files from the source image folders.
* run train.py as usual
* wait
* run merge_faces_larger.py on your image folder.

Directory structure for training data should look like (image names for example purposes only):

    .\data\A\alignments.json
    .\data\A\imageA1.jpg
    .\data\A\imageA2.jpg
    .\data\B\alignments.json
    .\data\B\imageB1.jpg
    .\data\B\imageB2.jpg



