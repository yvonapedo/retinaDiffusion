# This is the official repo of ["Enhancing Diffusion-Based Retina Segmentation Using Transformer and Entropy"](..).


The command for training:

## On Single GPU



```
python single_task_train.py
```

Change from multi_task_train.py to multi_task_test.py for testing.

If train and test on Windows, 

remove notation
```# os.environ["CUDA_VISIBLE_DEVICES"]='0'  # if train on windows```
in _multi_task_train.py_ and
```# args.dist_backend = 'gloo'  # if train on windows```
in _distribute_utils.py_. 

