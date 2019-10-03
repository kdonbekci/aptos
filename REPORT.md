## Goals: 
1. Clean up and normalize the noisy retina photos.
2. Create a model that predicts the [Diabetic Retinopathy](https://en.wikipedia.org/wiki/Diabetic_retinopathy) grade (on a scale of 5) from the photos. 

## Approach:
For this [Kaggle](https://www.kaggle.com/) competition, I followed a traditional data science approach. First step was the establish a baseline for the prediction task. After fitting some simple linear models, it became very clear that a convolutional approach was necessary in order to extract the signal from the input photos. From then onwards, I tried models that are well-established in literature such as [VGG](https://neurohive.io/en/popular-networks/vgg16/) and [EfficientNet](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html). An apparent problem (and a blessing) was that the photos differed greatly in terms of their image resolution, aspect ratios, and color profile. However, the blessing in disguise was that all of the eye images could be assumed to look *similar* as they had been originated from the same probability distribution. This meant that it was possible to exercise an aggressive data cleaning procedure, during which I normalized all of the aforementioned image properties. 

The second challenge was the fact that there was a major class imbalance in the dataset. To overcome this issue, I modified the loss functions I used to weigh underrepresented classes more in the training process. 


## Outcome:
After training and submitting, I was able to reach a respectable score in the competition. More importantly, I learned about the challenges working with medical data and became much more familiar with more traditional image processing methods that can enhance the quality of the images used for training. 