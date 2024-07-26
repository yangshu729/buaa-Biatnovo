# export CUDA_VISIBLE_DEVICES=5
# python v2/main.py --train --train_dir ~/v2/transformer_apid_decoder_two_model
# #!/bin/bash

# Set the CUDA device to GPU 5
export CUDA_VISIBLE_DEVICES=5

# Define the Python script and its arguments
PYTHON_SCRIPT="v2/main.py"
ARGS="--train --train_dir /root/v2/transformer_api_no_early_stop"

# Function to run the Python script
run_script() {
    python $PYTHON_SCRIPT $ARGS
}

# Loop to retry if the script is killed by signal 9 or 15
while true; do
    run_script
    EXIT_STATUS=$?

    if [ $EXIT_STATUS -eq 0 ]; then
        echo "Script executed successfully."
        break
    elif [ $EXIT_STATUS -eq 137 ]; then
        # 137 is the exit code for SIGKILL (128 + 9)
        echo "Script was killed by SIGKILL. Retrying..."
    elif [ $EXIT_STATUS -eq 143 ]; then
        # 143 is the exit code for SIGTERM (128 + 15)
        echo "Script was killed by SIGTERM. Retrying..."
    else
        echo "Script failed with exit status $EXIT_STATUS. Stopping."
        break
    fi

    # Optional: Wait for a few seconds before retrying
    sleep 5
done