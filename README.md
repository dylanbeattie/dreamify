# dreamify
A script for generating "deep dream" images, using Python and TensorFlow

### Prerequisites

Python 3 (I installed Anaconda from [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual) - this worked fine on Windows 10 and on macOS)

Tensorflow:

```bash
pip install --upgrade tensorflow
```

### Usage:

```bash
python dreamify.py
```

The script generates four iterations based on the sample image; each successive iteration increase the number of steps and octave scale used to produce a stronger effect:

````
0_image.jpg
1_image.jpg
2_image.jpg
3_image.jpg
````

