
## Environmental setup

#### Training Env

Step 1. Conda create from environments/laser_train_env.yml
Step 2. Install a patched version of [grounding dino](https://github.com/video-fm/GroundingDINO)
Step 3. Install a patched version [segment-anything-2](https://github.com/video-fm/video-sam2) 
Step 4. Install [Scallop language](https://github.com/scallop-lang/scallop) according to the instructions. 
Step 5. Verify setup by running the [train script](src/training/train_clip_distributed_restore.py)

#### Evaluation Env
Step 1. Conda create from environments/laser_eval_env.yml
Step 2. Install a patched version of [grounding dino](https://github.com/video-fm/GroundingDINO)
Step 3. Install a patched version [segment-anything-2](https://github.com/video-fm/video-sam2) 
Step 4. Verify setup by running the demo jupyter notebook

## Datasets

### Training Dataset Downloading

### Preprocessing

#### Video Mask Processing
```src/Preprocess/mask_generation.py```

#### STSL Generation
- Using GPT to generate JSON structures of the video captions. ```src/Preprocess/GPTSpecs_1.py```
- Parsing the generated structures to create STSL programs. ```src/Preprocess/GPTSpecs_2.py```
- Negative sample generation for contrastive learning. ```src/Preprocess/NegativeSampler.py```

## Common Questions

### 1. Question: My SAM2 shows post processing issues
Answer: Ensure your CUDA Tool kit and your pytorch has the same version. 

Take 12.4 as an example:
If you have sudo access, you can simply do  `sudo apt-get install cuda-toolkit-12-4`. If not, follow the instructions below.
- Download CUDA. You need to create an installation directory, to install without sudo access.
    ```bash
    # Install CUDA 12.4 without sudo
    # Download CUDA installer
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
    # Create installation directory
    mkdir -p ~/cuda-12.4
    # Run installer
    sh cuda_12.4.0_550.54.14_linux.run --toolkit --toolkitpath=~/cuda-12.4 --defaultroot=~/cuda-12.4 --no-opengl-libs --no-man-page --no-drm
    ```
- Once you run the installer, a UI interface will appear. Accept the end user license agreement. Then you will see a CUDA Installer menu. Note - replace the install path in the screenshots with the path of the installation directory you created.
cuda installer menu default
- Uncheck the checked Driver section. Navigate to Options using arrow keys, press Enter.
uncheck driver
- The Options menu will appear. Navigate to Toolkit Options.
cuda options menu
- In Toolkit options, navigate to Change Toolkit Install Path. Make sure your install path is the installation directory you created earlier.
cuda change toolkit install path
- After changing the toolkit install path, stay in the Toolkit Options menu. Make sure to uncheck "Create symbolic link from /usr/local/cuda". Navigate to Done.
cuda toolkit options menu
- Navigate to Library install path. Ensure that the install path is also the installation directory.
cuda library install path
- Navigate to Done. Then navigate to Install. After installing, set your environment variables.
```bash
 echo 'export PATH=/home/[user]/cuda/cuda-12.4/bin:$PATH' >> ~/.bashrc
 echo 'export LD_LIBRARY_PATH=/home/[user]/cuda/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
 source ~/.bashrc
```
- Verify your installation.
```bash
nvcc --version
```
- Install PyTorch support for CUDA 12.4
```bash
conda install pytorch=2.5.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
- Verify PyTorch and CUDA 12.4
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA toolkit: {torch.version.cuda}")
```

## Contributing
### Contributing Guidelines
1. Create a Github issue outlining the piece of work. Solicit feedback from anyone who has recently contributed to the component of the repository you plan to contribute to. Reach out for feedback on the ESCA slack. If it's adding a feature, please share a brief 1 page google document describing what you're adding and how you will implement it.
2. Checkout a branch from main - preferably name your branch [github username]/[brief description of contribution]
3. Create a pull request that refers to the created github issue in the commit message.
- To link to the github issue, in your commit for example you would simply add in the commit message:
    ```
    [what the PR does briefly] #[commit issue]
    ```
    Then when you push your commit and create your pull request, Github will automatically link the commit back to the issue. Add more details in the pull request, and request reviewers from anyone who has recently modified related code.
4. After 1-2 approvals, merge your pull request.

