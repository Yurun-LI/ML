# 机械学习

### 分类

- 无监督学习
  - 入力データのみで学習
  - クラスタリングなどに利用
  - データ分布などを把握するに利用

- 监督学习
  - 入力データと正解データを与える学習
  - クラス分類や回帰に利用
- 强化学习
  - ロボットなどの行動の学習
  - 行動に対する報酬を学習する
    - 報酬が高くなるように行動

### 无监督学习

- 分类问题(**Classification**)
- 回归问题(**Regression**)

#### **结构**

输入--->抽取特征--->函数--->输出

1. **模型**
   - 选择与数据和目标相符合的模型	
2. **评价基准**(误差函数)
   - 可以说是学习什么是好的连接的评价函数
   - 最小二乘误差, crossentropy
3. **最优化**
   - 探索使得评价函数最小的参数

## 目录

- ### 第一回	线性回归

- ### 第二回　多項式回帰

- 



##  第一回	线性回归

$$
y=f(x)=x^TW
$$

- #### 统计和数据挖掘

- #### 机械学习

#### 最小二乘法

**使得误差函数最小化--->最优化**
$$
L=\sum(f(x_i)-y_i)^2
$$
使得*L*最小化,求*f(x)*.

#### **泛化误差**

<img src="/Users/liyurun/Library/Application%20Support/typora-user-images/image-20201203140449195.png" alt="image-20201203140449195" style="zoom:33%;" />



#### 回归问题中经常使用到的误差函数(error function)

- ***Mean Square Error***	(MSE)
  - 误差平方和
  - 容易计算

$$
L(w)={1}/{N}\times|y_i-f(x_i)|
$$

- ***Mean Absolute Error***  (MAE)
  - 误差大小的平均

$$
L(w)=1/N\sum(y_i-f(x_i))
$$

- ***Root Mean Squared Error***  (RMSE)
  - MSE的平方根

$$
L(w)=1/N\sum|y_i-f(x_i)|	
$$

#### 最小二乘法的解法

##### **解析的な解き方**

- 線形モデルで利用
- 誤差関数が微分可能な下に凸の関数

<img src="/Users/liyurun/Library/Application%20Support/typora-user-images/image-20201203142532231.png" alt="image-20201203142532231" style="zoom:30%;" />

<img src="/Users/liyurun/Library/Application%20Support/typora-user-images/image-20201203142715306.png" alt="image-20201203142715306" style="zoom:30%;" />

##### **繰り返し演算による解き方**

- パーセプトロン
- 非線形の問題でも利用可能

<img src="/Users/liyurun/Library/Application%20Support/typora-user-images/image-20201203144239305.png" alt="image-20201203144239305" style="zoom:33%;" />

#### Python実現

```python
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.Xb = np.concatenate((np.ones(len(self.X)).reshape(-1, 1), self.X), axis=1)

    def J(self, theta):
        try:
            return np.sum((np.dot(self.Xb, theta.reshape(-1, 1)) - self.y.reshape(-1, 1)) ** 2) / len(self.Xb)
        except:
            return float('inf')

    def dJ(self, theta):
        res = np.empty(len(theta))
        res[0] = np.sum(self.Xb.dot(theta) - self.y)

        for i in range(1, len(theta)):
            res[i] = (self.Xb.dot(theta) - self.y).dot(self.Xb[:, i])
        return res / len(self.Xb)

    def GradientDescent(self, init_theta, n_iters, epsilon, alpha):
        theta = init_theta
        i_iter = 0

        while i_iter < n_iters:
            gradient = self.dJ(theta)
            last_theta = theta
            theta = theta - alpha * gradient

            if abs(self.J(theta) - self.J(last_theta)) < epsilon:
                break
            i_iter += 1
        return theta
    def

    def plotting(self, theta):
        Y = self.Xb.dot(theta.reshape(-1, 1))
        plt.scatter(self.X, self.y.reshape(-1, 1), marker='x', label='data')
        plt.plot(self.X, Y, color='red', linestyle='-', linewidth=2, label='Linear Regression')
        plt.grid(which='major', color='black', linestyle=':')
        plt.grid(which='minor', color='black', linestyle=':')
        plt.legend(loc='best')


####################################
####################################
def main():
    np.random.seed(444)
    x = 2 * np.random.random(size=100)
    y = x * 3. + 4. + np.random.normal(size=100)

    X = x.reshape(-1, 1)
    X_b = np.hstack((np.ones(len(X)).reshape(-1,1),X))
    INIT_THETA = np.ones(2)
    EPSILON = 1e-8
    N_ITERS = 1e4
    ALPHA = 0.01

    LR = LinearRegression(X, y)
    plt.plot(X,X_b.dot(INIT_THETA),color = 'green',linewidth= 2,label = 'Initial line')
    plt.plot(x, x * 3. + 4., color='orange',linewidth =2,label = 'true line' )
    theta = LR.GradientDescent(INIT_THETA, N_ITERS, EPSILON, ALPHA)
    LR.plotting(theta)
    plt.show()


if __name__ == "__main__":
    main()
```



## 第二回　多項式回帰・正則化

多項式を用いて線形回帰を拡張した非線形回帰と、非線形回帰にける過学習と多重共線性問題を抑える正規化方法について紹介する。

### 多項式回帰

#### 　回帰

- 観測されるデータにはノイズが乗っている
  $$
  y=f(\mathbf{x})+\epsilon
  $$
  <img src="https://raw.githubusercontent.com/Yurun-LI/images/PicGo/image-20201222134408632.png?token=AOMVBP6SYUEGN25EKFWKEYK74F45S" alt="image-20201222134408632" style="zoom:50%;" />

- 回归想做的是?

  - 从(**x**,y)的数据组中推断出f(**x**)
    - 对f(**x**)的假定(线性还是非线性)
    - 对f(**x**)的参数的推断

#### 线性回归

##### 因变量

- 一元因变量(特征值)的情况:
  $$
  f(x)=ax+b
  $$
  <img src="https://raw.githubusercontent.com/Yurun-LI/images/PicGo/image-20201222135227898.png?token=AOMVBP2KRU3VOGYKDVKJMLS74F54Y" alt="image-20201222135227898" style="zoom:50%;" />

- 多元因变量的情况:
  $$
  f(\mathbf{x})=w_0+w_1x_1+w_2x_2+...+w_mx_m
  $$

  $$
  \mathbf{x}=[x_1,x_2,x_3,...,x_m]^T
  $$

  <img src="https://raw.githubusercontent.com/Yurun-LI/images/PicGo/image-20201222135254033.png?token=AOMVBPZAYBIQGEHNVSOEOJS74F56M" alt="image-20201222135254033" style="zoom:55%;" />

##### 向量表示

$$
f(\mathbf{x_b^T})=\mathbf{w\cdot{}x_b^T}=w_0\cdot1+w_1x_1+w_2x_2+...
$$

$$
\mathbf{x_b^T}=[1,x_1,x_2,...x_m]^T
$$

#### 线性回归和非线性回归

##### 线性回归

形如:
$$
f(x)=ax+b
$$
二乘误差**MSE**等误差函数的情况下可以求得解析解

##### 非线性回归

形如:
$$
f(x) = bx/(a+x)
$$
一般无法求得解析解

多项式的情况下形如:
$$
f(\mathbf{x})=w_0+w_1x+w_2x^2+...
$$
<img src="https://raw.githubusercontent.com/Yurun-LI/images/PicGo/image-20201222141748154.png?token=AOMVBPYANKN6YJ3P7AT5J4C74GA32" alt="image-20201222141748154" style="zoom:50%;" />

#### 线性基底函数MODEL

##### MODEl

$$
f(\mathbf{x})=w_1\phi_1(\mathbf{x})+w_2\phi_2(\mathbf{x})+...
$$

##### 过拟合(overfitting)

随着多项式次数增加

- 对于学习data的Error变小
- 对于新data的推测能力变弱

即:泛化误差增大

##### 多重共线性(Multicolinearity)

- 多重共线性:特征量间的相关很高的情况

  - 预测值变得不稳定

    解析解:
    $$
    \mathbf{w}=(X^TX)^{-1}X^TY
    $$
    特征量XtX相关变高-->行列式的逆矩阵难于求解

    而对于行列式来说:

    ​		行列正则-->逆行列可计算

    ​		行列非正则-->无法正确计算

### 正则化

#### 抑制过拟合和多重共线性

- MODEL单纯化

  - 表现变差

- 对参数施加限制(正则化)

  - L2正则化

    1. 参数的L2范数变小

    2. 参数值变小

       特征的微小变化放大

  - L1正则化

    1. 参数的L2范数变小
    2. 多个参数值会变0

#### 使用正则化的回归

- **L2正则化:Ridge回归**
  $$
  L=\sum{(y_i-f(\mathbf{x}_i))^2}+\lambda||\mathbf{w}||^2_2
  $$

  $$
  ||\mathbf{w}||^2_2=\sum{w_j^2}
  $$

- 

- **L1正则化:LASSO**

$$
L=\sum{(y_i-f(\mathbf{x}_i))^2}+\lambda||\mathbf{w}||_1
$$

$$
||\mathbf{w}||_1=\sum{|\mathbf{w}_j|}
$$

- **L1 + L2 同时使用:ElasticNet**

$$
L=\sum{(y_i-f(\mathbf{x}_i))^2}+\lambda||\mathbf{w}||_1+\lambda||\mathbf{w}||^2_2
$$

#### 示意图

##### L2正则化

<img src="https://raw.githubusercontent.com/Yurun-LI/images/PicGo/image-20201222145603275.png?token=AOMVBP44KGBU5XAPA4KJZYC74GFLK" alt="image-20201222145603275" style="zoom:50%;" />

##### L1正则化

![image-20201222145637432](https://raw.githubusercontent.com/Yurun-LI/images/PicGo/image-20201222145637432.png?token=AOMVBP6FSUNIUVCA3GEG5I274GFNU)

#### Ridge回归的解析解

- **通常的最小二乘解**

$$
\mathbf{w}=(X^TX)^{-1}X^TY
$$

- **Ridge回归**

$$
\mathbf{w}=(X^TX+\lambda{}I)^{-1}X^TY
$$

​		正则化
$$
X^TX\to{}X^TX+\lambda{}I
$$

#### 如何评价MODEL的好坏?

- 基准(泛化误差generalization error)

  用为学习过的数据进行误差评价

  [什么是泛化误差?他和训练误差的区别?](https://zhuanlan.zhihu.com/p/33449363)

- 评价
  - [Hold out法](https://www.bookstack.cn/read/huaxiaozhuan-ai/spilt.5.e9c1eb9ad9c3265b.md)
  - [cross-validation法](https://zhuanlan.zhihu.com/p/24825503)

##### Hold out法

data = train data + test data

train data 用于学习

test data 用于评价

​	**评价data存在偏差的时候,难以准确评价** 

​			**增加data数量,可以减少偏差的影响**

<img src="https://raw.githubusercontent.com/Yurun-LI/images/PicGo/image-20201222152842945.png?token=AOMVBP7NNRJS4XKB6SUKDTK74GJFS" alt="image-20201222152842945" style="zoom:50%;" />

##### cross-validation法

k分割交叉验证(k-fold cross validation)

1. 将data分割成k个set

2. 其中1个set用于评价,其余用于学习

3. 改变02中评价set,重复进行k轮评价,用其结果的平均值来进行评价

   **信赖度高,但需要多轮学习和评价**

**Example**

k = 4

![image-20201222153529730](https://raw.githubusercontent.com/Yurun-LI/images/PicGo/image-20201222153529730.png?token=AOMVBP2G3O2EWNDFRLYGNQK74GJ7C)