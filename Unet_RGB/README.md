### Segmentation of wheat rust disease with Unet. <br>

To train a unet model with 3 channel(RGB) input image run **unet.py**. It segments rust part in the image. <br>

### Three step model testing

**multi_unet_testing.py** applies three step model on testing:

- Apply thresholding with multiple color spaces
- Pass patches to Resnet18 model as second layer of classification (The Resnet18 model can be pretrained with unsupervised learning, such as with Simsiam)
- Apply unet on the patches that are classified as rust by Resnet18 and thresholding. Image is given as RGB to the unet model
### Datset
You can access part of the trainig dataset from [**here**](https://drive.google.com/file/d/1RCWvtiNe1uqbDqEry8HsVoLh24eQqNyJ/view?usp=sharing) <br>
[**Link**](https://drive.google.com/file/d/1a0-uZvADu6q3S6FUCgb8I-AagqVK4Dj7/view?usp=sharing) for testing images
