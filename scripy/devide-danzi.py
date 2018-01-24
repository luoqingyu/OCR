#-*- coding: UTF-8 -*-

'''
1、读取指定目录下的所有文件
2、读取指定文件，输出文件内容
3、创建一个文件并保存到指定目录
'''
import os
import  random
import shutil
import PIL.Image as Image


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print path
        print  ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print path
        print  ' 目录已存在'
        return False





# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    hero_dir=[]
    danzi_list = []
    for allDir in pathDir:
        allDir =unicode(allDir, 'utf-8')
        child = os.path.join('%s%s' % (filepath, allDir))
        #print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        hero_dir.append(child)
        danzi_list.append(allDir)
    return  hero_dir,danzi_list

def eachFile1(filepath):
    dir_list = []
    name_list = []
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        name_list.append(allDir)
        child = os.path.join('%s%s' % (filepath+'/', allDir))



        dir_list.append(child)
    return  dir_list,name_list



if __name__ == '__main__':
    filePath,danzi_list = eachFile("/home/lqy/project/OCR/ocr_cnn_lstm_ctc/ocr_cnn_lstm_ctc/data/danzi/")
    for i in danzi_list:
        print  i
        path  = '/home/lqy/project/OCR/ocr_cnn_lstm_ctc/ocr_cnn_lstm_ctc/data/danzi-train/' +i
        mkdir(path)
        path = '/home/lqy/project/OCR/ocr_cnn_lstm_ctc/ocr_cnn_lstm_ctc/data/danzi-test/' + i
        mkdir(path)

    train_pic_dir=[]
    test_pic_dir=[]
    for i in filePath:
        pic_dir,pic_name=eachFile1(i)
        random.shuffle(pic_dir)
        train_list=pic_dir[0:int(0.7*len(pic_dir))]
        test_list=pic_dir[int(0.7*len(pic_dir)):]
        for j in train_list:
            fromImage = Image.open(j)
            j=j.replace('danzi','danzi-train')
            fromImage.save(j)
        for k in test_list:
            fromImage = Image.open(k)
            k=k.replace('danzi','danzi-test')
            fromImage.save(k)

















