*updated_lstm_for_reading_v1* - an lstm that uses updated centering and scaling, adds reading speed (true) and demographic features after the lstm unit. AUC 0.97-0.98. The not-cleaned-up but working version is *updated_lstm_for_reading_added_speed_works_not_cleaned_up*.

*updated_lstm_for_reading_v2* - an lstm that uses updated centering and scaling, adds predicted reading speed and demographic features after the lstm unit. Data preprocessor returns X, y, subj, and reading speed. AUC 0.93. 

*Speed_prediction_nestedCV* - an LSTM that predicts reading speed. It uses updated centering and scaling, adds demographic features after the lstm unit. Data preprocessor returns X, y, subj, and reading speed (!), this preprocessor would be useful in the final version of the enhanced LSTM that deals with x, y, subj, and reading speed.
This is a stand-alone training that I need to find the necessary parameter combination for the main training that joins two parts (*updated_lstm_for_reading_v2*).

*Multihead_fixation_prediction_v1.ipynb* - a transformer that predicts the properties of randomly masked fixations. I've tried several position encoders, CustomPositionalEncoding seems to be the best, and AbsolutePositionalEmbedding the worst - loss is almost 10x.
