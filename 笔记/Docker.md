# 安装docker报错Hardware assisted virtualization and data execution protection must be enabled in the BIOS

https://blog.csdn.net/mythest/article/details/92999646

**解决方法**

其实我这个应该算是 Hyper-V异常导致的，所以要么禁用之后再启用，要么直接运行以下命令,算是重启这个服务：

```
bcdedit /set hypervisorlaunchtype auto
```

之后再重启电脑就ok了，可以愉快地开始docker旅程了。

