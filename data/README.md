# Data Collection

## MNIST
 This is a dataset of 70,000 28x28 grayscale images of the 10 digits. The first 60,000 images are training data, and the remaining 10,000 images are testing data. More info can be found at the [MNIST homepage](http://yann.lecun.com/exdb/mnist/).

```
import numpy as np
X, Y = np.load('MNIST_X.npy'), np.load('MNIST_Y.npy')
assert X.shape == (70000, 28, 28)
assert Y.shape == (70000,)
```

## COIL20 (Columbia University Image Library)
This is a database of gray-scale images of 20 objects. The objects were placed on motorized turntable.The turn table was rotated through 360 degrees to vary object pose with respect to fixed camera.Images of objects were taken at pose interval of 5 degrees.This corresponds to 72 images per object.

