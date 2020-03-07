# American Sign Language (ASL) Fingerspelling

## Background

According to some statistics published by CSD, there are approximately **70 million** deaf-mute people around the world (**1%** of the world's population) who use sign language. Another report by WHO crunches the numbers to **466 million** people. While translation services have become easily accessible for about 100 languages, sign language is still an area that hasn’t been explored to the same extent. But even in this age of technology and communication, we are yet to see a universal translation system that helps bridge the gap between people that can and cannot speak.

With such a large demographic of people left in the dark, it seems imperative that a reliable translation system is set up. One that will aid in breaking the language barrier and allowing indiscriminate communication with all. The goal of this project is to detect and accurately translate the letters in American Sign Language (ASL). This can later be expanded to many different sign languages too.

>   #### J & Z
>
>   J & Z are the only letters in ASL that require motion. As a consequence, I decided to skip these letters as these would be difficult to detect. Henceforth, any mention of ASL means all letters from A-Z excluding J & Z unless otherwise specified.

## Plan Of Work

The project shall be implemented by using the power of machine learning. Using transfer learning on 3 well-known models, namely, MobileNet, ResNet, and Inception. These are convolutional neural networks which are used on images. The dataset used to train the model is a combination of multiple datasets comprising from online sources and self-created images. The goal is to translate the letters of ASL given as input through a webcam in real-time.

## Get Started

### Step 1: `create_mapping.py`

Helps to create a json file with the required labels.

```bash
usage: create_mapping.py [-h] [--string STRING] [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --string STRING, -s STRING
                        Create a mapping of letters from a string (default: ABCDEFGHIKLMNOPQRSTUVWXY)
  --output OUTPUT, -o OUTPUT
                        Path to output json file (default: ../data/mapping.json)
```

Define a string of labels for the dataset or leave blank for ASL. Then, run the following

```bash
$ python create_mapping.py -s [LABELS] -o [OUTPUT FILE]
```

For example

```bash
$ python create_mapping.py -s ABC -o mapping.json
$ cat mapping.json
{
    "A": 0,
    "B": 1,
    "C": 2
}
```

### Step 2: `config.py`

Tune the model parameters based on your application. The following parameters can be tweaked:

* `EPOCHS`: number of passes through the training set, _default = 64_
* `SHAPE`: the shape of square image, _default = 250_
* `CHANNELS`: number of color channels, _default = 3_
* `BATCH_SIZE`: number of training samples used in one iteration, _default = 128_
* `NUM_CLASSES`: number of categorical labels for classification, defined by `mapping` in [step 1](#step-1:-create_mapping.py)