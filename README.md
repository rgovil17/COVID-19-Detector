# COVID-19 Detector
A Convolutional Neural Network (CNN) trained to predict whether a person has coronavirus or not by looking at their posteroanterior (PA) chest X-ray.

### The Dataset
The Dataset used is hosted on my [Dropbox](https://www.dropbox.com/s/wu76tnt41przqhe/CovidDataset.zip).
The structure of the dataset is as follows:
 - Train (224 images)
   - Covid (112)
   - Normal (112)
 - Val (60 images)
   - Covid (30)
   - Normal (30)

The Covid Dataset has been taken from [this](https://github.com/ieee8023/covid-chestxray-dataset) GitHub repository.  
The Normal Dataset is part of [this](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset hosted on Kaggle. 


### Running the Python file
You can run the python file from the command prompt. Make sure to give an image path using '-i' argument.  
`python covid_detection.py -i xray.jpeg`

<img src="/src/img1.png" width=90% alt="Command prompt">

The X-ray image will be shown along with the result, as predicted by the model.

<img src="/src/img2.png" width=50% alt="Result">

### Why this model is not reliable?
Although the model shows almost 97% accuracy on the validation dataset, it cannot be considered reliable (yet?) because
 - The dataset is very small (Only 224 images for training)
 - We need an expert radiologist's supervision to evaluate our model
