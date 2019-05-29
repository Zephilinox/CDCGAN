# CDCGAN
Minecraft texture pack creator using a conditional deep convolutional generative adversarial network

This was developed for a university module.

main.py is the CDCGAN and it has a few command line options:

If you do not specify an option, it will train the CDCGAN from scratch.

If you specify "all", it will output 100 of every texture using CDCGAN-generator-test1600.h5

If you specify "minecraft", it will generate a texture pack using CDCGAN-generator-test1600.h5

If you specify a specific texture's label, it will generate that texture using CDCGAN-generator-test1600.h5

There's a sample texture pack, and within "texture-pack-extraction" you will find the extraction process for getting training data and some existing data.

## 1600 Epochs

### Sample Output
![](https://github.com/Zephilinox/CDCGAN/blob/master/output-test/1600.png)

### Discriminator Loss
![](https://github.com/Zephilinox/CDCGAN/blob/master/output-test/1600%20-%20d_loss.png)

### Discriminator Accuracy
![](https://github.com/Zephilinox/CDCGAN/blob/master/output-test/1600%20-%20accuracy.png)

### Generator Loss
![](https://github.com/Zephilinox/CDCGAN/blob/master/output-test/1600%20-%20g_loss.png)

### Sample Input
![](https://github.com/Zephilinox/CDCGAN/blob/master/output-test/1600-input.png)

