# Keras 之 LSTM 有状态模型（stateful LSTM）和无状态模型（stateless LSTM）

参考资料：https://blog.csdn.net/qq_27586341/article/details/88239404

 **目录**

[1. 区别](https://blog.csdn.net/qq_27586341/article/details/88239404#1. 区别)

[2. 例子](https://blog.csdn.net/qq_27586341/article/details/88239404#t0)

[3. 疑问解答](https://blog.csdn.net/qq_27586341/article/details/88239404#t1)

[4. 实战](https://blog.csdn.net/qq_27586341/article/details/88239404#t2)

[  1. 实例1：官方的example——lstm_stateful.py](https://blog.csdn.net/qq_27586341/article/details/88239404#t3)

[  2. 实例2：用Keras实现有状态LSTM——电量消费预测](https://blog.csdn.net/qq_27586341/article/details/88239404#t4)

[  3. 实例3：用Keras实现有状态LSTM序列预测](https://blog.csdn.net/qq_27586341/article/details/88239404#t5) 

[普通多层神经网络](https://blog.csdn.net/qq_27586341/article/details/88239404#t6)

[stateless LSTM](https://blog.csdn.net/qq_27586341/article/details/88239404#t7)

[单层Stateful LSTM](https://blog.csdn.net/qq_27586341/article/details/88239404#t8)

[双层stacked Stateful LSTM](https://blog.csdn.net/qq_27586341/article/details/88239404#t9)

------

有状态的RNN，能在训练中维护跨批次的状态信息，即为当前批次的训练数据计算的状态值，可以用作下一批次训练数据的初始隐藏状态。因为Keras RNN默认是无状态的，这需要显示设置。stateful代表除了每个样本内的时间步内传递，而且每个样本之间会有信息(c,h)传递，而stateless指的只是样本内的信息传递。

### 参考目录：

- [官方文档](https://keras.io/zh/getting-started/faq/#how-can-i-use-stateful-rnns)
- [Stateful LSTM in Keras](https://link.jianshu.com/?t=http%3A%2F%2Fphilipperemy.github.io%2Fkeras-stateful-lstm%2F) （必读圣经）
- [案例灵感来自此GitHub](https://link.jianshu.com/?t=https%3A%2F%2Fgithub.com%2Fsachinruk%2FPyData_Keras_Talk)
- [Stateful and Stateless LSTM for Time Series Forecasting with Python](https://link.jianshu.com/?t=https%3A%2F%2Fmachinelearningmastery.com%2Fstateful-stateless-lstm-time-series-forecasting-python%2F) 
- [时间序列数据生成器（TimeseriesGenerator）](https://www.jianshu.com/p/4466f64007fd)
- [深度学习之路（一）：用LSTM网络做时间序列数据预测](https://www.jianshu.com/p/6b874e49b906)

------

# **1. 区别**

- **stateful LSTM**：

​    能让模型学习到你输入的samples之间的时序特征，适合一些长序列的预测，哪个sample在前，哪个sample在后对模型是有影响的。

​    优点：更小的网络，或更少的训练时间。

​    缺点：需要使用反应数据周期性的批大小来训练网络，并在每个训练批后重置状态。

- **stateless LSTM：**

​    输入samples后，默认就会shuffle，可以说是每个sample独立，之间无前后关系，适合输入一些没有关系的样本。

------

# 2. 例子

1. stateful LSTM：我想根据一篇1000句的文章预测第1001句，每一句是一个sample。我会选用stateful，因为这文章里的1000句是有前后关联的，是有时序的特征的，我不想丢弃这个特征。利用这个时序性能让第一句的特征传递到我们预测的第1001句。（batch_size = 10时）
2. stateless LSTM：我想训练LSTM自动写诗句，我想训练1000首诗，每一首是一个sample，我会选用stateless LSTM，因为这1000首诗是独立的，不存在关联，哪怕打乱它们的顺序，对于模型训练来说也没区别。

**当使用有状态 RNN 时，假定：**

1. 所有的批次都有相同数量的样本
2. 如果 x1 和 x2 是连续批次的样本，则x2[i]是 x1[i] 的后续序列，对于每个i。

**要在 RNN 中使用状态，需要:**

1. 通过将 batch_size 参数传递给模型的第一层来显式指定你正在使用的批大小。例如，对于 10 个时间步长的 32 样本的 batch，每个时间步长具有 16 个特征，batch_size = 32。
2. 在 RNN 层中设置 stateful = True。
3. 在调用 fit() 时指定 shuffle = False。

**重置累积状态：**

1. 使用 model.reset_states()来重置模型中所有层的状态

2. 使用layer.reset_states()来重置指定有状态 RNN 层的状态

   ------

    

# **3. 疑问解答**

1. **将一个很长的序列（例如时间序列）分成小序列来构建我的输入矩阵。那LSTM网络会发现我这些小序列之间的关联依赖吗？**
       不会，除非你使用 stateful LSTM 。大多数问题使用stateless LSTM即可解决，所以如果你想使用stateful LSTM，请确保自己是真的需要它。在stateless时，长期记忆网络并不意味着你的LSTM将记住之前batch的内容。
2. **在Keras中stateless LSTM中的stateless指的是?**
       注意，此文所说的stateful是指的在Keras中特有的，是batch之间的记忆cell状态传递。而非说的是LSTM论文模型中表示那些记忆门，遗忘门，c，h等等在同一sequence中不同timesteps时间步之间的状态传递。
       假定我们的输入X是一个三维矩阵，shape = （nb_samples, timesteps, input_dim），每一个row代表一个sample，每个sample都是一个sequence小序列。X[i]表示输入矩阵中第i个sample。步长啥的我们先不用管。
       当我们在默认状态stateless下，Keras会在训练每个sequence小序列（=sample）开始时，将LSTM网络中的记忆状态参数reset初始化（指的是c，h而并非权重w），即调用model.reset_states()。
3. **为啥stateless LSTM每次训练都要初始化记忆参数?**
       因为Keras在训练时会默认地shuffle samples，所以导致sequence之间的依赖性消失，sample和sample之间就没有时序关系，顺序被打乱，这时记忆参数在batch、小序列之间进行传递就没意义了，所以Keras要把记忆参数初始化。
4. **那stateful LSTM到底怎么传递记忆参数？**
       首先要明确一点，LSTM作为有记忆的网络，它的有记忆指的是在一个sequence中，记忆在不同的timesteps中传播。举个例子，就是你有一篇文章X，分解，然后把每个句子作为一个sample训练对象（sequence），X[i]就代表一句话，而一句话里的每个word各自代表一个timestep时间步，LSTM的有记忆即指的是在一句话里，X[i][0]第一个单词（时间步）的信息可以被记忆，传递到第5个单词（时间步）X[i][5]中。
       而我们突然觉得，这还远远不够，因为句子和句子之间没有任何的记忆啊，假设文章一共1000句话，我们想预测出第1001句是什么，不想丢弃前1000句里的一些时序性特征（stateless时这1000句训练时会被打乱，时序性特征丢失）。那么，stateful LSTM就可以做到。
       在stateful = True 时，我们要在fit中手动使得shuffle = False。随后，在X[i]（表示输入矩阵中第i个sample）这个小序列训练完之后，Keras会将将训练完的记忆参数传递给X[i+bs]（表示第i+bs个sample）,作为其初始的记忆参数。bs = batch_size。这样一来，我们的记忆参数就能顺利地在sample和sample之间传递，X[i+n*bs]也能知道X[i]的信息