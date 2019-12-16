# Layer-sequential unit-variance (LSUV) initialization for tf.keras

This is sample code for LSUV and initializations, implemented in python script within Keras framework.

LSUV initialization proposed by Dmytro Mishkin and Jiri Matas in the article [All you need is a good Init](https://arxiv.org/pdf/1511.06422.pdf) consists of the two steps. 
 - First, pre-initialize weights of each convolution or inner-product layer with orthonormal matrices. 
 - Second, proceed from the first to the final layer, normalizing the variance of the output of each layer to be equal to one.

Original implementation can be found at [ducha-aiki/LSUVinit](https://github.com/ducha-aiki/LSUVinit).

## Result Comparison
|  | Default Init | LSUV Init |
|---------------|--------------|-----------|
| Fashion-MNIST | 83.15 % | 85.65 % |

### References
 - [ducha-aiki/LSUV-keras](https://github.com/ducha-aiki/LSUV-keras)
 - Notebook modified from [kashif/tf-keras-tutorial/blob/tf2/5-conv.ipynb](https://github.com/kashif/tf-keras-tutorial/blob/tf2/5-conv.ipynb)