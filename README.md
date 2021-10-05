# BlindBot-Walking Assistance for Visually Impaired
It is a Deep Learning based project for visually impaired peoples. Visually impaired people face a lot of problems just in their day to day activities. Doing basic chores is a challenge in itself. In this project, we aim to provide a walking aid by making them aware of their surroundings through the help of audio devices such as headphones.
I have identified 3 main tasks in this problem:
1. Image Captioning
2. Object Detection
3. Text to Speech

Firstly, It will generate a one line caption to the image which can describe the image to the blind. To do this task we have used deep learning techniques i.e. Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). A general description of the image will help the blind to understand the activities happening around him/her.

Secondly, We need to correctly detect and classify the objects around them. We have used the YOLO for object detection. We have used a Tensorflow model for this task. By this detecting the objects in the image we can also decide that to which side of the person they are located. This will help the blind to walk accordingly.

Lastly, whatever nearby objects that have been detected and the caption i.e. description of the image is generated need to be delivered to the user in the form of the audio. This audio will be played to user through headphones.
