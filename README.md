# Convolutional Neural Networks Classification of Lung Tissue Images 
## Introduction
### Lung Tumors
A lung tumor is a tumor that occurs in the lung tissue itself or in the airways that lead to the lungs. Lung tumors can be either cancerous (malignant) or benign (non-cancerous). Benign tumors are not cancerous, so they will not spread to other parts of the body. They grow slowly, or might even stop growing or shrink, and they are usually not life-threatening and do not need to be removed. 

- **Benign**

  ![lungn40](https://user-images.githubusercontent.com/73894812/118424545-6bce5a00-b695-11eb-9235-7d1dcd5f32e9.jpeg)

For cancerous tumors, there are two main types, and the one we will focus on are non-small cell lung cancer (NSCLC).
About 80% to 85% of lung cancers are NSCLC. The main subtypes of NSCLC are adenocarcinoma, squamous cell carcinoma, and large cell carcinoma. These subtypes, which start from different types of lung cells are grouped together as NSCLC because their treatment and prognoses (outlook) are often similar. The two subtypes of NSCLC we will focus on are adenocarcinoma and squamous cell carcinoma.
- **Adenocarcinoma**

  ![lungaca54](https://user-images.githubusercontent.com/73894812/118418565-df1c9f80-b686-11eb-845f-4890993200bb.jpeg)


  Adenocarcinomas start in the cells that would normally secrete substances such as mucus. This type of lung cancer occurs mainly in current or former smokers, but it is also the most common type of lung cancer seen in non-smokers. It is more common in   women than in men, and it is more likely to occur in younger people than other types of lung cancer.
  Adenocarcinoma is usually found in the outer parts of the lung and is more likely to be found before it has spread.

- **Squamous Cell Carcinoma**

  ![lungscc24](https://user-images.githubusercontent.com/73894812/118418645-3589de00-b687-11eb-8474-b0f7895d4e09.jpeg)

  Squamous cell carcinomas start in squamous cells, which are flat cells that line the inside of the airways in the lungs. They are often linked to a history of    smoking and tend to be found in the central part of the lungs, near a main airway (bronchus).

### Histopathology
Histopathology refers to the microscopic examination of tissue in order to study the manifestations of disease. Specifically, in clinical medicine, histopathology refers to the examination of a biopsy or surgical specimen by a pathologist, after the specimen has been processed and histological sections have been placed onto glass slides. The [histopathological image dataset](https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images) we will use is created by [Andrew A. Borkowski et al](https://arxiv.org/abs/1912.12142v1) (2019). The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and augmented to 15,000 using the Augmentor package.

### Convolutional Neural Networks
In recent years, neural networks have become one of the best-performing artificial-intelligence systems in many fields. neural networks combine many nonlinear functions to model the complicated relationship between input features and output labels. They are applied in areas such as computer vision, speech recognition, natural language processing, character recognition, signature verification, and many others. With the utilization of large data sets and high computing power, neural networks can adapt to varying conditions and have demonstrated high accuracy in various tasks. The convolutional neural networks (CNNs) are primarily used in image processing and recognition. The convolutional layer, the pooling layer, and the fully connected layer are three basic building blocks of CNNs.

## Methods
In this project, we will use the Keras package to construct four CNN structures: DenseNet169, ResNet50, ResNet50V2, and VGG16. There are two files of images in the dataset: train and test. We will first split the train data into training and validation sets by a ratio of 0.1 with "keras.preprocessing.image.ImageDataGenerator" function. We will also conduct data prerpocessing by feeding parameters to this function. We will set each sample mean to 0, rescale the data by 1/255 so that each value is between 0 and 1, use a shear range and zoom range of 0.2, and horizontally flip the images.

```python
train_dataGenerator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
                                                                   rescale=1./255,
                                                                   shear_range=0.2,
                                                                   zoom_range=0.2,
                                                                   horizontal_flip=True,
                                                                   validation_split=0.1)
```

Then we will set the batch size to 128 and target size to 64 by 64.

```python
train = train_dataGenerator.flow_from_directory("/content/lung_image_sets/train", 
                                                class_mode='categorical', 
                                                batch_size=128, 
                                                target_size=(64, 64),
                                                subset='training')
```

There are 540 images in the test set, 13014 in the training set, and 1446 in the validation set.

Finally, we will train the models with 10 epochs and an early stopping using validation loss and patience of 3.

```python
model = DenseNet169(inputShape=(64,64,3), outputClasses=3, accMetrics=['categorical_accuracy'])
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train, epochs=10, callbacks=[callback], verbose=1, validation_data=valid)
```

### CNN Structures
First we will load the structures from "keras.applications" function, and we will apply transfer learning and use weights trained on [ImageNet](https://www.image-net.org/). We will also use ADAM as the optimizer for each CNN in the project.

#### DenseNet169
There are four versions of the [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) and we will use DenseNet169. Below are the structures of each DenseNet. 

![Screen Shot 2021-05-16 at 9 12 41 PM](https://user-images.githubusercontent.com/73894812/118419996-a3d09f80-b68b-11eb-9563-95790656ed2d.png)

```python
def DenseNet169(inputShape, outputClasses, accMetrics):
    m = keras.models.Sequential()
    m.add(keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=inputShape))
    for layer in m.layers[:16]:
        layer.trainable = False
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(512, activation='relu'))
    m.add(keras.layers.Dropout(0.5))
    m.add(keras.layers.Dense(3, activation='softmax'))
    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-4,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              metrics=accMetrics)

    return m
```

#### ResNet50

There are five versions of the [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) and we will use ResNet50. Below are the structures of each ResNet. 

![Screen Shot 2021-05-16 at 9 20 42 PM](https://user-images.githubusercontent.com/73894812/118420435-d6c76300-b68c-11eb-9991-83f32a948685.png)

```python
def ResNet50(inputShape, outputClasses, accMetrics):
    m = keras.models.Sequential()
    m.add(keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=inputShape))
    for layer in m.layers[:16]:
        layer.trainable = False
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(512, activation='relu'))
    m.add(keras.layers.Dropout(0.5))
    m.add(keras.layers.Dense(3, activation='softmax'))
    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-4,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              metrics=accMetrics)

    return m
```

#### ResNet50V2

There are three versions of the [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) and we will use ResNet50V2. 

```python
def ResNet50V2(inputShape, outputClasses, accMetrics):
    m = keras.models.Sequential()
    m.add(keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=inputShape))
    for layer in m.layers[:16]:
        layer.trainable = False
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(512, activation='relu'))
    m.add(keras.layers.Dropout(0.5))
    m.add(keras.layers.Dense(3, activation='softmax'))
    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-4,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              metrics=accMetrics)

    return m
```
#### VGG16

There are six versions of the [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1603.05027) and we will use VGG16. Below are the structures of each VGG. 

![Screen Shot 2021-05-16 at 9 42 46 PM](https://user-images.githubusercontent.com/73894812/118421813-d8def100-b68f-11eb-9099-187a6f1476ac.png)

```python
def VGG16(inputShape, outputClasses, accMetrics):
    m = keras.models.Sequential()
    m.add(keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=inputShape))
    for layer in m.layers[:16]:
        layer.trainable = False
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(512, activation='relu'))
    m.add(keras.layers.Dropout(0.5))
    m.add(keras.layers.Dense(3, activation='softmax'))
    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-4,beta_1=0.9,beta_2=0.999,epsilon=1e-8),
              metrics=accMetrics)

    return m
```
## Results
### DenseNet169
It was stopped after epoch 8.

![densenet169 acc](https://user-images.githubusercontent.com/73894812/118423275-a97db380-b692-11eb-9645-4552358ddc9a.png)
![densenet169 loss](https://user-images.githubusercontent.com/73894812/118421963-396e2e00-b690-11eb-96b0-0a5123470569.png)

```python
model.evaluate(test)
```

5/5 [==============================] - 5s 891ms/step - loss: 0.2524 - categorical_accuracy: 0.8981

[0.2523741126060486, 0.8981481194496155]

### ResNet50

![resnet50 acc](https://user-images.githubusercontent.com/73894812/118423316-bef2dd80-b692-11eb-820d-1f1b39b753a7.png)
![resnet50 loss](https://user-images.githubusercontent.com/73894812/118422397-23ad3880-b691-11eb-9333-c00379ca9532.png)

```python
model.evaluate(test)
```

5/5 [==============================] - 5s 912ms/step - loss: 0.7891 - categorical_accuracy: 0.6185

[0.7890522480010986, 0.6185185313224792]

### ResNet50V2

![resnet50v2 acc](https://user-images.githubusercontent.com/73894812/118423329-c4e8be80-b692-11eb-8bfc-7c5aaa3176a8.png)
![resnet50v2 loss](https://user-images.githubusercontent.com/73894812/118422504-59522180-b691-11eb-8df2-a7d49647e5b1.png)

```python
model.evaluate(test)
```

5/5 [==============================] - 5s 901ms/step - loss: 0.3052 - categorical_accuracy: 0.8833

[0.30521562695503235, 0.8833333253860474]

### VGG16

![vgg16 acc](https://user-images.githubusercontent.com/73894812/118423336-cb773600-b692-11eb-9255-9dcf1577008c.png)
![vgg16 loss](https://user-images.githubusercontent.com/73894812/118422598-84d50c00-b691-11eb-95e6-2dac3117518a.png)

```python
model.evaluate(test)
```

model.evaluate(test)
5/5 [==============================] - 5s 1s/step - loss: 0.3549 - categorical_accuracy: 0.8648

[0.3548664152622223, 0.864814817905426]

### Ranking 
1. DenseNet169--89.81%
2. ResNet50--88.33%
3. VGG16--86.48%
4. ResNet5--61.85%

## Conclusion
In this project, we investigate the classification performance of four CNN models: DenseNet169, ResNet50, ResNet50V2, and VGG16. Out of these four, DenseNet169 had the best performance of achieving an 89.81% accuracy. ResNet50V2 and VGG16 also yield comparable results of 88.33% and 86.48%. Surprisingly, ResNet50 reached an accuracy of only 61.85%. The application of highly accurate CNNs in clinical medicine could improve the diagnosing accuracy and reliability and shorten the diagnosing time. However, since CNNs work like a gray box, the ethnical delimma of relying on machines for making diagnosis would have to be discussed before using it on a greater scale.
