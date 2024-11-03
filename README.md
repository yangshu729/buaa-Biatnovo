### Prerequisites for developing
1. Install Git LFS.

    Install Git LFS and its related packages by running the following commands.
    ```bash
    sudo apt-get update -y
    sudo apt-get install git-lfs
    ```
    After installation, navigate to the repository directory and execute `git lfs install` to enable Git LFS.

2. Install depedencies
    ```bash
    git clone https://github.com/yangshu729/buaa-Biatnovo.git
    cd buaa-Biatnovo
    pip3 install -r requirements.txt
    ```

3. Build lib from srouce
    ```bash
    cd DataProcess/DataProcess
    python3 deepnovo_cython_setup.py build && python3 deepnovo_cython_setup.py install
    ```

    To ensure that the package has been installed correctly, go the to repo root folder, verify it by running a simple test using the following command: `python3 Biatnovo/DataProcessing/deepnovo_worker_test.py`

4. Knapsack data preparing

    Unzip the `knapsack.npy.zip`, and put the uncompressed data into the root folder of the repository.

### The steps to run the predict code

#### Start inference

1. Download the model weight.
```bash
export RELEASE_VERSION="v0.1"
wget -O sbatt_deepnovo.pth "https://github.com/yangshu729/buaa-Biatnovo/releases/download/${RELEASE_VERSION}/sbatt_deepnovo.pth"
wget -O spectrum_cnn.pth "https://github.com/yangshu729/buaa-Biatnovo/releases/download/${RELEASE_VERSION}/spectrum_cnn.pth"
mkdir -p /root/checkpoints
mv sbatt_deepnovo.pth /root/checkpoints
mv spectrum_cnn.pth /root/checkpoints
```

2. Download test feature/MGF files and run command:
```bash
export DENOVO_INPUT_DIR=/root/input_data
# You need to download/prepare feature (features.csv)/MGF (spectrum.mgf) files and put them into the DENOVO_INPUT_DIR
export DENOVO_OUTPUT_DIR=/root/outputs
export DENOVO_OUTPUT_FILE=output.csv
mkdir -p $DENOVO_INPUT_DIR
mkdir -p $DENOVO_OUTPUT_DIR
bash ./predict_v2.sh
```
to start inference.
