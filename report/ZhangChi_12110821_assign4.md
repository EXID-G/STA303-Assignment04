# Artificial Intelligence-Homework 4

**Name：** 张弛（ZHANG Chi）

**SID：** 12110821

**Link of pull request: **



## Introduction

In this assignment,  we are asked to make the conformal prediction of self-selected models by using the `TorchCP` library. For my work, I use 2 models (*ResNet18* and *DenseNet121*) and 2 datasets (*FashionMNIST* and  *CIFAR-100*) to test the performance of 4 CP scores and 3 predictors. 

- There are 3 code files in my pull request. 
	- data_prepare.py: to download and prepare the data. The function `load_data` returns the training data, calibration data, and test data by giving the specified dataset name.
	- model_prepare.py: to download and prepare the model. The function `training_model` can help train the model on a specified dataset and the default epoch is 20.
	- ass4.ipynb: the main file to test the performance. I try all permutations and combinations of 4 CP scores and 3 predictors.

- Datasets
	- The **FashionMNIST dataset** comprises 60,000 grayscale images of Zalando fashion items, with 10,000 additional test images. **With 10 distinct categories** such as T-shirt and Dress,     the task involves classifying each image into one of these fashion classes. 
	- In contrast, the **CIFAR-100 dataset** presents a more complex challenge, featuring 60,000 color images with **100 fine-grained categories** grouped into 20 superclasses. Each class has 500 training images and 100 testing images. CIFAR-100 requires classifying images into diverse categories, offering a broader range of objects for recognition compared to the fashion-focused FashionMNIST dataset. 
	- These datasets stand as benchmarks for image classification tasks, each contributing unique characteristics to assess the performance of machine learning models.

- Model: I use the pre-trained models **ResNet18** and **DenseNet121** and retrain them using both two datasets. For each model, I modify its last linear layer so that it can work normally on new data sets and tasks. Therefore, there are a total of 4 models for following procedures, e,g. `densenet121_cifar100, densenet121_fashionmnist, resnet18_cifar100, resnet18_fashionmnist`.

	Here is the code of the model loading function in `model_prepare.py`.

	```python
	## model_prepare.py
	
	def load_resnet18(num_classes):
	    # 加载预训练的 ResNet18 模型
	    resnet18 = torchvision.models.resnet18(pretrained=True)
	
	    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
	    return resnet18
	
	def load_densenet(num_classes):
	    # 加载预训练的 DenseNet 模型
	    densenet = torchvision.models.densenet121(pretrained=True)
	
	    # 修改最后一层的全连接层
	    in_features = densenet.classifier.in_features
	    densenet.classifier = nn.Linear(in_features, num_classes)
	    
	    return densenet
	```



##  Experiment

### Experiment preparation

Conformal prediction (CP) is a framework in machine learning and statistics that provides confidence estimates for individual predictions made by a model.  In the following part, I will introduce 4 score functions and  3 predictors, respectively:

- CP scores:
	- **THR (Threshold Conformal Predictors): **THR is designed for generating confidence intervals based on statistical confidence levels. It is grounded in statistical principles, providing a reliable assessment of prediction confidence and suitability for classification tasks.
	- **APS (Adaptive Prediction Sets):** APS is an adaptive prediction set method that considers the adaptability and flexibility of prediction sets. By adaptively adjusting the prediction set, it enhances robustness and accommodates diverse data features. APS excels in addressing changes and complexities in data.
	- **SAPS (Sorted Adaptive Prediction Sets):** SAPS is a sorted adaptive prediction set method that improves performance by sorting the prediction set. Sorting aids in better understanding the importance and impact of samples, making the prediction set more interpretable. SAPS is particularly useful in scenarios where the sorting of samples is needed. **The weight is set to 0.2.** 
	- **RAPS (Regularized Adaptive Prediction Sets):** RAPS is an improvement over APS, introducing regularization to enhance performance. Regularization helps prevent overfitting and improves the algorithm's generalization on various datasets. RAPS may have advantages in handling large-scale, high-dimensional datasets and enhancing prediction robustness. **Here I set the penalty as 1.**
- predictors:
	- **ClusterPredictor** is a class-conditional conformal predictor designed to handle situations with many classes. The method likely involves techniques for efficient prediction and calibration within a large number of class labels.
	- **ClassWisePredictor** is a method designed for multi-class classification. It focuses on class-conditional conformal prediction, offering confidence levels for predictions in each class. It is particularly suitable in scenarios where distinguishing between multiple classes is crucial.
	- **SplitPredictor**

The accuracy of two models in two datasets’ testing dataloaders is shown in the table below (epoch = 30). It is more difficult to classify images into 100 categories.

|             | Cifar100  (100 classes) | Fashionmnist  (10 classes) |
| :---------: | :---------------------: | :------------------------: |
|  ResNet18   |         42.95%          |           85.32%           |
| DenseNet121 |         49.66%          |           83.91%           |

###  Experiment  Results



Conclusion



## Appendix



## Reference

[https://github.com/ml-stat-Sustech/TorchCP](https://github.com/ml-stat-Sustech/TorchCP)









