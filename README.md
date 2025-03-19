## Installation
### Create a conda environment
```bash
conda create -n isaac-vr python=3.8
conda activate isaac-vr
pip install setuptools==65.5.0
pip install --upgrade pip wheel==0.38.4
pip install termcolor
```

### IsaacGym
Download the IsaacGym installer from the [IsaacGym website](https://developer.nvidia.com/isaac-gym) and follow the instructions to download the package by running (also refer to the [FurnitureBench installlation instructions](https://clvrai.github.io/furniture-bench/docs/getting_started/installing_furniture_sim.html#download-isaac-gym)):

- Click "Join now" and log into your NVIDIA account.
- Click "Member area".
- Read and check the box for the license agreement.
- Download and unzip `Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release`.

Once the zipped file is downloaded, move it to the desired location and unzip it by running:

```bash
tar -xzf IsaacGym_Preview_4_Package.tar.gz
```


Now, you can install the IsaacGym package by navigating to the `isaacgym` directory and running:

```bash
pip install -e python --no-cache-dir --force-reinstall
```

_Note: The `--no-cache-dir` and `--force-reinstall` flags are used to avoid potential issues with the installation we encountered._

_Note: Please ignore Pip's notice that `[notice] To update, run: pip install --upgrade pip` as the current version of Pip is necessary for compatibility with the codebase._

_Tip: The documentation for IsaacGym  is located inside the `docs` directory in the unzipped folder and is not available online. You can open the `index.html` file in your browser to access the documentation._

You can now safely delete the downloaded zipped file and navigate back to the root directory for your project. 

### Install packages
   ```
   conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install -c conda-forge rospkg catkin_pkg 
   pip install -U einops matplotlib scikit-learn rosinstall_generator wstool rosinstall six vcstools msgpack empy cadquery pymeshlab pandas tqdm easydict timm opencv-python h5py
   ```