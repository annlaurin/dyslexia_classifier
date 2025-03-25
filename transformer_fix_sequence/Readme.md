## Ehnahnced LSTM: Reading speed prediction
- To identify the best hyperparameters for reading speed prediction and to assess model performance, run 'experiment_reading_speed.py'
- To generate predicted reading speed scores and train an enhanced LSTM that uses the predicted scores as input, run 'enhanced_lstm.py'

## Transformer model
- To identify the best hyperparameters for predicting masked fixations and to assess model performance, run 'experiment.py'
- To fine-tune the pretrained model in order to predict dyslexia label:
	- for fine-tuning on a fully frozen pretrained model, run 'experiment_fine-tuning.py --model "transformer_tuning_frozen"'
	- for fine-tuning on a partially trainable model (2 fully-connected layers of the upper feed-forward network are trainable), run 'experiment_fine-tuning.py --model "transformer_tuning_partial"'
	- for fine-tuning on a fully trainable model, run 'experiment_fine-tuning.py --model "transformer_tuning_learnable"'
