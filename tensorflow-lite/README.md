## Tensorflow Lite Model Converter
### IoT models easily deployed from your .h5 weights file

Simply specify: 

1. Input Model 
2. Output Model 

## How It Works

Upon the start-up the application takes parameters and converts the input model to optimized tensorflow lite version for edge device usage

## Running

Running the application with the `-h` option yields the following usage message:
```
python3 convert.py -h
```
The command yields the following usage message:
```
usage: convert.py [-h] -m MODEL -i INPUT
                                      

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        output model will have tflite appended
  -i INPUT, --input INPUT
                        input model will have .h5 appended
 
example 
python3 convert.py \
       --model=my_lite_model\
       --input=current_model
```

![alt text](https://github.com/vvagias/cnvrg_ai_library_extras/blob/main/tensorflow-lite/tf_lite.png?raw=true)
