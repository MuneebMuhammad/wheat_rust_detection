# SimSiam in PyTorch
This is an implementation of the self-supervised learning algorithm SimSiam in PyTorch.<br>

### Set up
Place patches training path that will be used in training.<br>
place patches in validation path that will be used for testing.<br>
Training and validation path is given in data.py<br>
Run **wheat_train.py** to train the model. <br>
The loss of the model is recorded in a csv file, which location can be set in train.py file.<br>
If the loss immidiately goes to -1 then the model has collapsed; however if the loss slowly decreases to -1 then the model is learning.<br>
After training Simsiam, you can use the weights in downstream tasks by fine tuning the model.

### Dataset
[**Link**](https://drive.google.com/file/d/1BZejxepZj0yFwzXgjQz7OSqzukKf7BHg/view?usp=sharing) for dataset
