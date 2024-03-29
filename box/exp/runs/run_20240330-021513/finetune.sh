SCRIPT_PATH="box/exp/run.py"

# Set the path to your data file
DATA_PATH="box/data/asr_results.json"
PREDICT_PATH="box/data/asr_results.json"
MODEL="t5-base"

# Run the Python script with the specified arguments
OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node $1 $SCRIPT_PATH \
    --output_dir=$2 \
    --num_train_epochs=100 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --save_steps=500 \
    --eval_steps=500 \
    --logging_dir="./logs" \
    --data_path=$DATA_PATH \
    --predict_path=$PREDICT_PATH \
    --model_name_or_path=$MODEL 