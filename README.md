# Emoji-Recognition
Real Time Hand Gesture Recognition (Emoji Recognition), implemented with keras and opencv on python, to detect upto 10 gestures using a Convolutional Neural Network . With the use of the webcam each frame was taken and subtracted from the background to identify the hands and then passed through the CNN model to predict the emoji. An overall precision of 96% was achieved.

The dataset was created using dataset.py and is available at https://www.kaggle.com/gsnikkitha/real-time-hand-gesture-recognition.

EmojiRecognitionModel.ipynb can be used to train the model. The trained model is stored in model.hdf5.

Finally RealTimeEmoji.ipynb can be used to predict the hand recognition in real time.
