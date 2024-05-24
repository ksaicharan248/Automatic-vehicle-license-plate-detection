Automatic vehicle license plate detection plays an important role in various applications such as law enforcement and traffic management and toll collection. In this project, we proposed a method for automatic license plate detection using machine learning techniques. The proposed algorithm consists of several steps. In the first step the camera captures images of vehicles on the road. These images are then processed by our model. Once the license plate region is identified, then it will be cropped using one of the python library called Python Image Library (PIL). Subsequently Gaussian Blur is applied to reduce noise in the cropped license plate image. To increase accuracy the cropped image will be resized. Then it is fed to the model which uses CNN for character detection. By analyzing the x and y positions of each character, the model will identifies the individual characters on the license plate. The recognized characters are stored in an array based on their accuracy, and subsequently sorted based on their positions within the license plate. Finally the sorted characters are extracted, providing a robust and accurate method for automatic license plate detection. The experimental results define the effectiveness and accuracy of the proposed method. This means it’s really good at recognizing license plates in pictures of vehicles. This could make a big difference in how we manage traffic and help police do their jobs better.


# Input samples :

![image](https://github.com/ksaicharan248/PROJECT-2024/assets/83259388/f3485c52-5417-40ee-b901-fc825aa0d0a4)

# License Plate detection
![image](https://github.com/ksaicharan248/PROJECT-2024/assets/83259388/dda15f53-e0b3-4cf0-a1d7-8f8e5d3927f4)


# Optical chara Recoginition
![image](https://github.com/ksaicharan248/PROJECT-2024/assets/83259388/63786536-0fcc-47be-884f-8f2b0d757d01)





