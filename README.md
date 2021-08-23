# Vision Transformers Pipeline with PyTorch Lightning
First install [PyTorch-Pretrained-ViT repository](https://github.com/arkel23/PyTorch-Pretrained-ViT) by cloning it then `pip install -e .`
Train with `python train.py --arguments`

Detailed argument list plus all the ones included with PyTorch Lightning Trainer class:
```
usage: train.py [-h] [--seed SEED] [--no_cpu_workers NO_CPU_WORKERS] [--results_dir RESULTS_DIR]
                [--save_checkpoint_freq SAVE_CHECKPOINT_FREQ] [--dataset_name {cifar10,cifar100,imagenet}] [--dataset_path DATASET_PATH]
                [--deit_recipe] [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE] [--optimizer {sgd,adam}]
                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--warmup_steps WARMUP_STEPS]
                [--model_name {B_16,B_32,L_16,L_32}] [--pretrained_checkpoint] [--checkpoint_path CHECKPOINT_PATH] [--transfer_learning]
                [--load_partial_mode {full_tokenizer,patchprojection,posembeddings,clstoken,patchandposembeddings,patchandclstoken,posembeddingsandclstoken,None}]
                [--interm_features_fc] [--conv_patching] [--logger [LOGGER]] [--checkpoint_callback [CHECKPOINT_CALLBACK]]
                [--default_root_dir DEFAULT_ROOT_DIR] [--gradient_clip_val GRADIENT_CLIP_VAL]
                [--gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES]
                [--num_processes NUM_PROCESSES] [--devices DEVICES] [--gpus GPUS] [--auto_select_gpus [AUTO_SELECT_GPUS]]
                [--tpu_cores TPU_CORES] [--ipus IPUS] [--log_gpu_memory LOG_GPU_MEMORY]
                [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE] [--overfit_batches OVERFIT_BATCHES]
                [--track_grad_norm TRACK_GRAD_NORM] [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]]
                [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS]
                [--max_steps MAX_STEPS] [--min_steps MIN_STEPS] [--max_time MAX_TIME] [--limit_train_batches LIMIT_TRAIN_BATCHES]
                [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES]
                [--limit_predict_batches LIMIT_PREDICT_BATCHES] [--val_check_interval VAL_CHECK_INTERVAL]
                [--flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS] [--log_every_n_steps LOG_EVERY_N_STEPS]
                [--accelerator ACCELERATOR] [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION]
                [--weights_summary WEIGHTS_SUMMARY] [--weights_save_path WEIGHTS_SAVE_PATH]
                [--num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--truncated_bptt_steps TRUNCATED_BPTT_STEPS]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler PROFILER] [--benchmark [BENCHMARK]]
                [--deterministic [DETERMINISTIC]] [--reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
                [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]] [--auto_lr_find [AUTO_LR_FIND]]
                [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]] [--terminate_on_nan [TERMINATE_ON_NAN]]
                [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]] [--prepare_data_per_node [PREPARE_DATA_PER_NODE]] [--plugins PLUGINS]
                [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL] [--distributed_backend DISTRIBUTED_BACKEND]
                [--move_metrics_to_cpu [MOVE_METRICS_TO_CPU]] [--multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE]
                [--stochastic_weight_avg [STOCHASTIC_WEIGHT_AVG]]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           random seed for initialization
  --no_cpu_workers NO_CPU_WORKERS
                        CPU workers for data loading.
  --results_dir RESULTS_DIR
                        The directory where results will be stored
  --save_checkpoint_freq SAVE_CHECKPOINT_FREQ
                        Frequency (in epochs) to save checkpoints
  --dataset_name {cifar10,cifar100,imagenet}
                        Which dataset to use.
  --dataset_path DATASET_PATH
                        Path for the dataset.
  --deit_recipe         Use DeiT training recipe
  --image_size IMAGE_SIZE
                        Image (square) resolution size
  --batch_size BATCH_SIZE
                        Batch size for train/val/test.
  --optimizer {sgd,adam}
  --learning_rate LEARNING_RATE
                        Initial learning rate.
  --weight_decay WEIGHT_DECAY
  --warmup_steps WARMUP_STEPS
                        Warmup steps for LR scheduler.
  --model_name {B_16,B_32,L_16,L_32}
                        Which model architecture to use
  --pretrained_checkpoint
                        Loads pretrained model if available
  --checkpoint_path CHECKPOINT_PATH
  --transfer_learning   Load partial state dict for transfer learningResets the [embeddings, logits and] fc layer for ViT
  --load_partial_mode {full_tokenizer,patchprojection,posembeddings,clstoken,patchandposembeddings,patchandclstoken,posembeddingsandclstoken,None}
                        Load pre-processing components to speed up training
  --interm_features_fc  If use this flag creates FC using intermediate features instead of only last layer.
  --conv_patching       If use this flag uses a small convolutional stem instead of single large-stride convolution for patch
                        projection.
```
