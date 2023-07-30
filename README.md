### Prerequisites for developing
1. Install git lfs: `sudo apt-get update -y && sudo apt-get install git-lfs` to install the related package. Then run `git lfs install` under the repo.

### The steps to run the predict code
1. Install the python-C library
 GO to Bianovo/CDataProcessing. run `python deepnovo_cpython_setup.py build && python deepnovo_cpython_setup.py install`. This command will comile the code and install the related library into global python python.
 To make make you installed the package correct. Please run `python Biatnovo/DataProcessing/deepnovo_worker_test.py`