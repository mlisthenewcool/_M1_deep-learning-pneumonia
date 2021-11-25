1. # Feature extraction and prediction

2. In this step we use a pre-trained neural network called VGG-19 as a feature extractor. This network is pre-trained on the ImageNet, who is composed with five blocks of convolutional layers and three fully connected layers. The detailed structure of VGG-19 is shown as follow:

3. ![]()

4. Each layer of the convolutional layers represent a certain level of graphic feature. In general, the more convolutional layers we have, the more complicated features can be represented. For example, the first convolutional layer may detect borders of different directions, and the next layer may detect all the combinations of borders. In this way, the last convolutional layer may be able to represent very complicated image features. These features are send to several fully connected layers to produce more complicated features and finally make the prediction. The feature of different layers extracted from a image of a cancer cell are shown as follows:

5. ![]()

6. Previous work has shown that features of different scales images extracted from block5_pool layer of VGG-19 can have good result with linear kernel SVM.  The sizes of images are (224+320*n, 224+320*n), where n is the scale, which is among 0, 1 and 2. An average pooling and the L2 normalization are applied to the features. An SVM are then used for the classification. In our work, we want to know the impact of changing layer of feature extraction and different pooling methods. We tried max pooling and average pooling on the features extracted from last three layers of VGG-19: block5_pool, fc1 and fc2.

7. VGG-19 is designed for the images of size (224,224). To get features correctly of different image scales, the last two layers of fully connected must be changed to convolutional layers. The weights of fc1 and fc2 trained from ImageNet are then used as a filter to be applied to the features. This is therefore a fully convolutional neural network. For example, the neural network structure when image scale is two is shown as follow:

8. ![]()

9. The datasets we used in this step is as before: miniMIT, chest_xray and kvasir_v2. The results are in the table below:

10. ![]()

11. 

12. # PCA and VLAD

13. To improve the classification of images, some methods based on image vector representation are also often used. According to the **Identity document classification as an image classification problem**, 

14. 

15. > This paper addressed the problem of identification documents classification as an image classi-
    > fication task. Several image classification methods are evaluated. We show that CNN features
    > extracted from pre-trained networks can be successfully transferred to produce image descriptors
    > which are fast to compute, compact, and highly performing. 

16. > technique of fine-tuning to adjust the model’s weights to adapt with the new images. in parallel there are
    > also other approaches based on Image vector representation to improve the classification of images, and
    > these techniques had a lot of success, with the results obtained for the Identity documents classification
    > using BOW,VLAD and Fisher vectors (Ronan Sicre and Furon, 2017) and (Jégou et al., 2010) In this