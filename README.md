### Prerequisites for developing
1. Install Git LFS.
    
    Install Git LFS and its related packages by running the following commands.
    ```bash
    sudo apt-get update -y
    sudo apt-get install git-lfs
    ```
    After installation, navigate to the repository directory and execute `git lfs install` to enable Git LFS.

2. Install blobfuse2.

    To install blobfuse2, Please refer to the documentation on [How to install blobfuse2](https://learn.microsoft.com/en-us/azure/storage/blobs/blobfuse2-how-to-deploy?tabs=Ubuntu#how-to-install-blobfuse2). After installing blobfuse2, use the following command to mount the related blob to the local system:
    ```bash
    mkdir ./biatnovo && blobfuse2 mount ./biatnovo --config-file=./blob_config.yaml
    ```

### The steps to run the predict code
#### Install the python-C library

Navigate to the "Bianovo/CDataProcessing" directory and execute the following command: `python deepnovo_cpython_setup.py build && python deepnovo_cpython_setup.py install`. This command will compile the native I/O related code and install the `DataProcess` library into the global Python path.

To ensure that the package has been installed correctly, you can verify it by running a simple test using the following command: `python Biatnovo/DataProcessing/deepnovo_worker_test.py`

### Knapsack data preparing
Unzip the `knapsack.npy.zip`, and put the uncompressed data into the root folder of the repository.

#### Start inference
 
Run command: 
```bash
python Biatnovo/predict.py --model ckpt_path --predict_spectrum testing_example.spectrum.mgf --predict_feature testing_example.feature.csv --cuda
```
to start inference.
