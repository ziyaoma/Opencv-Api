import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("otsu.jpg",0)

#blur = cv2.GaussianBlur(img,(5,5),0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])
histnorm = (hist-hist.min())/(hist.max()-hist.min())
#hist_norm = hist.ravel()/hist.max()
bins = np.arange(256)


plt.plot(histnorm)
plt.xlim([0,256])
plt.show()

fnmin = 99999
tmp = 0
fnmax = 0
tm = 0
p = hist/sum(hist)
histnorm = np.squeeze(histnorm)
for i in range(1,256):
    s1 = np.sum(histnorm[:i])
    s2 = np.sum(histnorm[i:])
    q1 = histnorm[i:]*bins[i:]
    m1 = np.sum(histnorm[:i] * bins[:i])/s1
    m2 = np.sum(histnorm[i:] * bins[i:])/s2
    mg = np.sum(histnorm* bins)/(s1+s2)

    v1 = np.sum((bins[:i] - m1)*(bins[:i] - m1)*(histnorm[:i])/s1)
    v2 = np.sum((bins[i:] - m2)*(bins[i:] - m2)*(histnorm[i:])/s2)
    fn = v1+v2
    if fn<fnmin:
        fnmin = fn
        tmp = i

    fm = (s1*(m1-mg)*(m1-mg)+ s2*(m2-mg)*(m2-mg))/(s1+s2)
    if fm>fnmax:
        fnmax = fm
        tm = i

ret1,th1 = cv2.threshold(img,tm, 255, cv2.THRESH_BINARY)
print("最大类间方差阈值：",tm,ret1)
ret2,th2 = cv2.threshold(img,tmp, 255, cv2.THRESH_BINARY)
print("最小类内方差阈值：",tmp,ret2)
ret3,th3 = cv2.threshold(img,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print("cv2.THRESH_OTSU阈值：",ret3)

cv2.imshow("th1",th1)
cv2.imshow("th2",th2)
cv2.imshow("THRESH_OTSU",th3)
cv2.waitKey(0)
