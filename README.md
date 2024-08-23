To train the models on an HPC (High-Performance Computing) cluster, you can use the provided shell scripts:

```bash
sbatch train_paragraphs.sh
sbatch train_sentences.sh
```

To run each model individualy you can run 
```bash
python3 ./code/basic.py \
--model-name <MODEL_NAME> \
--output-dir <OUTPUT_DIRECTORY> \
--batch-size <BATCH_SIZE> \
--num-epochs <NUM_EPOCHS> \
--learning-rate <LEARNING_RATE> \
--data-dir <DATA_DIRECTORY> \
--token-len <TOKEN_LENGTH> &> <LOG_FILE> &
```