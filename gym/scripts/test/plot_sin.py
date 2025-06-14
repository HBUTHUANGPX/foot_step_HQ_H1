import numpy as np
import matplotlib.pyplot as plt

# 生成x轴的数据，范围从0到2π，共1000个点
time = 2
dt = 0.02

x = np.zeros(int(time/dt))
t = np.zeros(int(time/dt))
print(x.shape[0])
for i in range(x.shape[0]):
    x[i]=1/35*i
    t[i]=dt*i
# 计算对应的y轴数据，即sin(x)的值
p = 2*np.pi*x
eps = 0.2

y = np.sin(p)/np.sqrt(np.sin(p)**2 + eps ** 2)

# 设置图片清晰度
# plt.rcParams['figure.dpi'] = 300

# 绘制sin曲线
plt.plot(t, y, label='sin(x)')

# 添加标题和坐标轴标签
plt.title('Sine Curve')
plt.xlabel('x')
plt.xticks(rotation=45)
plt.ylabel('y')

# 添加网格线
plt.grid(True)

# 添加图例
plt.legend()

# 显示图形
plt.show()