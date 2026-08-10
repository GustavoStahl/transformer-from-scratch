# Transformer from scratch

Implementation of the Transformer architecture, introduced in the paper: ["Attention is all you need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). This architecture was designed for the task of text translation.

 The present codebase implements:
- The transformer architecture
- A basic tokenizer to convert words into numbers
- Dataset and dataloader for the [WMT 2024 Chat task](https://wmt-chat-task.github.io/2024/)
- The "text translation using a transformer" training script

# Requirements
- tqdm
- NumPy
- pandas
- PyTorch
- TensorBoard

# Instructions
Clone the repository:
```bash
git clone --recurse-submodules <repo-url>
```
Or init the submodules in case you have cloned without it:
```bash
git submodule update --init --recursive
```
Install the required python packages:
```bash
# tested with python 3.10
pip install requirements.txt
```
Start the training process:
```bash
python train.py
```
On another terminal, launch tensorboard:
```bash
tensorboard --logdir=runs
```
Access the training visualizer at: http://localhost:6006.

> Skip training with tensorboard by adding `TB_MODE=disabled` before launching the training script. Example: `TB_MODE=disabled python train.py`.
