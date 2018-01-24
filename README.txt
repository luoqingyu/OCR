1.把制作好的数据存到./data 文件夹下，里面不要包括任何子文件夹  命名格式label.png  例:12看是0我.png

2.python mkdic.py 制作label的字典映射

3.一些配置参数在utils.py文件里，直接去文件里面改参数就行，没必要在运行的时候通过 --config 去指定
	a.'gpu_idex' 参数设置成你想跑得GPU的编号，例如: '0'  0号GPU运行程序
	b.'batch_size' 我默认128 你也可以根据情况自己改
4.运行代码：python main.py


