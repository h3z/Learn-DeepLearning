# Learn-DeepLearning
随便写一些实现来练习

# 学习记录
## LeNet-5
看deeplearning.ai学习时候，只知道大概，看这篇论文很多优缺点和理念更清晰了。（再结合吴恩达采访他的视频，和这篇深度学习简史的文章[神经网络和深度学习简史（全） - 云+社区 - 腾讯云](https://cloud.tencent.com/developer/article/1106685)）
1. 全连接网络对于处理图片的问题缺点是：参数多、失去了数据本身的拓扑属性。
2. 全连接网络那样， 如果一个图片发生偏移，对应的参数会有很多变化。反过来说，如果希望全连接网络可以适应图形变化，那需要很多参数应对变化引起的数据变化，意味着参数中很多都是重复的。（不知道是不是真这么回事，但是确实是一个思路）
3. 卷积网络通过权重复用，减少了参数数量，意味着可以更深，并且有利于提取局部特征，而且能适应图片的变换。
4. 知道了feature map这个词，感觉名字起的很好。


## Kaggle Digit Recognizer
1. 通过看kaggle digit recoginizer里note，知道了2个3 * 3的卷积层的输出等于一个5 * 5
2. 会用了LearningRateScheduler和ImageDataGenerator。
3. 在gcp上开了gpu。
4. 学会了Ensembling
5. 思考了自己的改进方案，最终果然发现🙅‍♂️


## AlexNet
没学到什么新东西…… 大概是复习吧。LeNet是提出了卷积网络的概念，AlexNet是从实践角度把这个理念成功运用于百万图片分为一千个类别。
1. 通过dropout和data augmentation来降低overfitting
2. 使用relu 替代当时使用比较多的tanh
3. 使用GPU提高效率
然后通过这些优化，成功拿了好成绩，惊呆了图像识别领域众人（在那个领域很多人都不知道卷积网络是什么）
