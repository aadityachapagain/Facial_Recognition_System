# Real Time Attendence System Using Face Recognition

The idea is to build application for a real-time attendence system using face detection and recognition.

## Prerequisites

* Tensorflow
* Numpy
* PIL
* Sklearn

## Project Examples

![](./examples/image_1.png)

![](./examples/image_2.png)


## Project Implementation

### Face Detection Using MTCNN

For Face Detection Purpose I used MTCNN network which composed of three cascaded network P-Net, R-Net and O-Net.And to supress the Overlapping Region of Faces I used NMS ( Non-Maximum Supression ) Algorithm to avoid overlapping Region which Represent same Face.

![https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf](./examples/MTCNN.png)


### Face Embeddings Generation Using Res-Net 

I used Res-Net for Embeddings Generation of input image which will generate 512 features embeddings of user face which is used to Recognize the Person.

### Using Triplet Loss ( Siamese Network ) for Face Recognition

So, first I used Triplet Loss to recognize the faces of each trained Individual but deu to some disadvantages of network like when Network is huge for small part of task like Face Recognition and when data's grow larger including trained faces more than like 200 it will take huge amount of time since it works by calculating distance between Anchor image and every trained image in database. Hence this methods needs lots of computation processing when trained faces grow larger. Well Most Important advantage of this method is that this is most accurate method.

### Using Machine learning Classifier
So, I used SVM and RandomForrest machine learning model to train the faces with corresponding labels and used in Face Recognition tasks.

## Applications
* Access and Security
* Criminal Identification
* Healthcare
* Payments
* Photo management Apps


## ToDos
- [x] Face Detection
- [x] face Classification
- [x] Building Classifier
- [X] Real Time face Recognition
- [x] Facial Expression Recognition
- [X] Using Database to Store User when detected with timestamp
- [ ] Integrating Face Recognition and Facial Expression Recognition
- [ ] check if detection can be done faster, if so re-implement it (optimize MTCNN for inference?)
- [ ] Implementing the whole system in browser
- [ ] Making system such that model can be directly updatable from browser