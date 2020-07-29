import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(image):
    pil_img = Image.fromarray(np.uint8(image))
    pil_img.show()


(x_train, y_train),  (x_test, y_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = y_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
img_show(img)
