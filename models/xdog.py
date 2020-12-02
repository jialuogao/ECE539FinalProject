import cv2
import numpy as np

def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
    img1 = cv2.GaussianBlur(img,size,sigma)
    img2 = cv2.GaussianBlur(img,size,sigma*k)
    return (img1-gamma*img2)

def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
    img = dog(img,sigma=sigma,k=k,gamma=gamma)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if(img[i,j] < epsilon):
                img[i,j] = 1
            else:
                img[i,j] = (1 + np.tanh(phi*(img[i,j])))
    return img

def xdog_thresh(img, sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1,alpha=1):
    img = xdog(img,sigma=sigma,k=k,gamma=gamma,epsilon=epsilon,phi=phi)
    #cv2.imshow("1",np.uint8(img))
    mean = np.mean(img)
    max = np.max(img)
    img = cv2.GaussianBlur(src=img,ksize=(0,0),sigmaX=sigma*3)
    #cv2.imshow("2",np.uint8(img))
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if(img[i,j] > mean):
                img[i,j] = max
    #cv2.imshow("3",np.uint8(img))
    return img/max

if __name__ == '__main__':
    # Open image in grayscale
    #img = cv2.imread('imgs/lena.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    img = cv2.imread('./imgs/scenery.png',cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img = cv2.resize(img,(400,400))
    print(img.shape)
    # k = 1.6 as proposed in the paper
    k = 1.6

    #cv2.imshow("Original in Grayscale", img)
    
    #cv2.imshow("Edge DoG",edge_dog(img,sigma=0.5,k=200, gamma=0.98))
    
    #cv2.imshow("XDoG GaryGrossi",np.uint8(xdog_garygrossi(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10)))

    #cv2.imshow("XDoG Project 1",np.uint8(xdog(img,sigma=0.4,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))
    cv2.imshow("orig",img)
    cv2.imshow("thres",np.uint8(255*xdog_thresh(img,sigma=2.2,k=1.6, gamma=0.98,epsilon=-0.1,phi=200)))

    #cv2.imshow("XDoG Project 2",np.uint8(xdog(img,sigma=1.6,k=1.6, gamma=0.5,epsilon=-1,phi=10)))

    # Natural media (tried to follow parameters of article)
    #cv2.imshow("XDoG Project 3 - Natural Media",np.uint8(xdog(img,sigma=1,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))

    #cv2.imshow("XDoG Project 4 - Hatch",np.uint8(hatchBlend(img)))

    cv2.waitKey(0)