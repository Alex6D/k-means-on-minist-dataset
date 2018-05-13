#coding:utf-8
from scipy.cluster.vq import *
from pylab import *
from scipy import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/',one_hot=False)

k = 10  #聚类的个数
number = 4  #聚类的数字

def get_data(number,image,label,total_num):
    data_num = 0
    for i in range(total_num):
        if label[i] == number:
            im = image[i].reshape([1,784])
            if data_num == 0:
                pre_data = im
                data_num+=1
            else:
                pre_data = np.vstack((pre_data,im))
                data_num+=1
    return pre_data,data_num

def k_means(data,k):
    pred_data = whiten(data)
    centroids,distortion = kmeans(pred_data,k)
    code,distance = vq(pred_data,centroids)
    return code,distance

def show_kmeans(data,code,k):
    for i in range(k):
        ind = where(code==i)[0]
        figure()
        gray()
        for j in range(minimum(len(ind),40)):
            subplot(4,10,j+1)
            imshow(data[ind[j]].reshape([28,28]))
            axis('off')
        savefig("%sth class.png" % str(i+1))
        print ("%sth image saved" %str(i+1))
    show()
    
def show_center(data,code,distance,k):
    for i in range(k):
        ind = where(code==i)[0]
        figure()
        gray()
        for j in range(3):
            subplot(2,2,j+1)
            min=ind[0]
            for i in range(len(ind)):
                if distance[ind[i]] < distance[min]:
                    min=ind[i]
            imshow(data[min].reshape([28,28]))
            distance[min]=99999
            axis('off')
        savefig("%sth class_max.png" % str(i+1))
        print ("%sth image saved" % str(i+1))
    show()
    
data,num = get_data(number,mnist.train.images,mnist.train.labels,54999)
print ("Have got %d samples of number:%d" % (num,number))
code,distance = k_means(data,k)
print ("k-means have done")
print "images saving..."
show_kmeans(data,code,k)
print "Everything done"
