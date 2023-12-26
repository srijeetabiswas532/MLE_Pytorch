[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/vYQ4W4rf)
# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py

Module 3.4 GPU vs. FastOps Wall clock run times:
* ![Alt text](/3_4_runtimes.jpg?raw=true "3.4 Runtimes")

Simple Dataset (Small Model):
* CPU:
* ![Alt text](/CPU_Simple.jpg?raw=true "Simple")
* GPU:
* ![Alt text](/GPU_Simple.jpg?raw=true "Simple")

Split Dataset (Small Model):
* CPU:
* ![Alt text](/CPU_Split.jpg?raw=true "Split")
* GPU:
* ![Alt text](/GPU_Split.jpg?raw=true "Split")

XOR Dataset (Small Model):
* CPU:
* ![Alt text](/CPU_xor.jpg?raw=true "XOR")
* GPU:
* ![Alt text](/GPU_xor.jpg?raw=true "XOR")


Simple Dataset (Large Model):
* CPU:
* ![Alt text](/CPU_large_Simple.jpg?raw=true "Simple")
* GPU:
* ![Alt text](/GPU_large_Simple.jpg?raw=true "Simple")

Split Dataset (Large Model):
* CPU:
* ![Alt text](/CPU_large_Split.jpg?raw=true "Split")
* GPU:
* ![Alt text](/GPU_large_Split.jpg?raw=true "Split")

XOR Dataset (Large Model):
* CPU:
* ![Alt text](/CPU_large_xor.jpg?raw=true "XOR")
* GPU:
* ![Alt text](/GPU_large_xor.jpg?raw=true "XOR")


