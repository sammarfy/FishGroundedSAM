# FishGroundedSAM
This repository is for fish background removal using GroundedSAM.

# :hammer_and_wrench: Install 

## Setting up a virtual environment
Follow the following code to setup the virtual environment.
```bash
python -m venv groundedsam
source groundedsam/bin/activate
pip install torch
```
Now, you should set the GroundingDINO separately.

## Installing GroundingDINO first

**Note:**

0. If you have a CUDA environment, please ensure the environment variable `CUDA_HOME` is set. It will be compiled under CPU-only mode if no CUDA is available.

Please make sure to follow the installation steps strictly; otherwise, the program may produce the following: 
```bash
NameError: name '_C' is not defined
```

If this happens, please reinstall the groundingDINO by reclone the git and do all the installation steps again.
 
#### How to check cuda:
```bash
echo $CUDA_HOME
```
If it prints nothing, then it means you haven't set up the path/

Run this so the environment variable will be set under the current shell. 
```bash
export CUDA_HOME=/path/to/cuda-11.3
```

Notice the cuda version should be aligned with your CUDA runtime, for there might exist multiple cuda at the same time. 

If you want to set the CUDA_HOME permanently, store it using:

```bash
echo 'export CUDA_HOME=/path/to/cuda' >> ~/.bashrc
```
After that, source the bashrc file and check CUDA_HOME:
```bash
source ~/.bashrc
echo $CUDA_HOME
```

In this example, /path/to/cuda-11.3 should replace the path where your CUDA toolkit is installed. You can find this by typing **which nvcc** in your terminal:

For instance, 
If the output is /usr/local/cuda/bin/nvcc, then:
```bash
export CUDA_HOME=/usr/local/cuda
```
**Installation:**

1. Clone the GroundingDINO repository from GitHub.

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

2. Change the current directory to the GroundingDINO folder.

```bash
cd GroundingDINO/
```

3. Install the required dependencies in the current directory.

```bash
pip install -e .
```

4. Download pre-trained model weights.

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
## Installing LangSAM to finish the installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
```
Delete this line from pyproject.toml file.

```bash
groundingdino = {git = "https://github.com/IDEA-Research/GroundingDINO.git"}
```
Once you are inside the `lang-segment-anything/` folder, run the following to finish the installation:

```bash
pip install -e .
```

**Adding the groundedsam env to jupyter kernel:**


```bash
pip install ipykernel
python -m ipykernel install --user --name=groundedsam
```

# :arrow_forward: Demo

Sample python code to run the GroundedSAM

```python
text_prompt, BOX_THRESHOLD = "fish", 0.30
model = LangSAM()
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold=BOX_THRESHOLD)  
```

Run the following code for background removal:

```bash
python background-removal.py \
-d input \
-i INHS_FISH_005052.jpg \
-o output
```
