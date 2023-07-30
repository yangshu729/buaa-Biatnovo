### Prerequisites for developing
1. Install git lfs: `sudo apt-get update -y && sudo apt-get install git-lfs` to install the related package. Then run `git lfs install` under the repo.
2. Install blobfuse2, please refer to this doc ti install [How to install blobfuse2](https://learn.microsoft.com/en-us/azure/storage/blobs/blobfuse2-how-to-deploy?tabs=Ubuntu#how-to-install-blobfuse2). After install, run `mkdir ./data && mount ./biatnovo --config-file=./blob_config.yaml` to mount the related blob to local.

### The steps to run the predict code
#### Install the python-C library

Go to Bianovo/CDataProcessing. run `python deepnovo_cpython_setup.py build && python deepnovo_cpython_setup.py install`. This command will comile the code and install the related library into global python python.
To make make you installed the package correct. Please run `python Biatnovo/DataProcessing/deepnovo_worker_test.py`

#### Start inference
 
Run command: 
```bash
python Biatnovo/predict.py --model ckpt_path --predict_spectrum testing_example.spectrum.mgf --predict_feature testing_example.feature.csv --cuda
```
to start inference.
