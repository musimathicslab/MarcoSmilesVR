# MarcoSmilesVR


<p align="center"><img src="./README_IMAGES/MSVRlogo.png" width="40%" height="40%">

## Project Overview

MarcoSmiles was conceived in 2016 within the Laboratory of Musimatics at the University of Salerno,
with the ambitious aim of exploring new techniques of human-computer interaction in the context of musical performances. This innovative platform is characterised by the possibility to fully customise the performance of a virtual musical instrument through the natural movement of the hands. MarcoSmiles integrates a MIDI management module, allowing the system to be used with any device equipped with a MIDI interface.

The integration of virtual reality provides MarcoSmiles with a significant advantage,
transforming the musical experience into an immersive experience. 
Users can have the opportunity to experiment with music in an immersive three-dimensional environment,
which attempts to reproduce a recording studio as faithfully as possible, where hand movements become the means to explore and shape one's own musical creation.
This fusion of technology and virtual reality amplifies human-machine interaction,
enabling a deeper connection between the user and the system. In this context, 
virtual reality is not just an upgrade but a transformation that elevates MarcoSmiles to a higher 
level of musical expression and opens up different scenarios for improvement.

<video width="100%" height="100%" controls autoplay>
  <source src="./README_IMAGES/3Denv.mp4" type="video/wp4">
 
</video>

## New Features
In previous versions, one or more _Leap Motion_ was used to capture hand information. 
However, this dynamic has changed in the VR version, where the use of different technologies 
related to the augmented reality visor, including the **integrated hand tracking functionality**, 
enables a new approach. Now, instead of using an additional device such as the Leap Motion, 
**the system makes use of the hand tracking subsystem already built into the visor**.

Another fundamental evolution concerns the **restructuring of the neural network** used, 
with a specific focus on the **type of learning implemented**. In the current version, 
we have abandoned the traditional approach to embrace a more advanced model: **reinforcement learning,
with a special emphasis on the implementation of Q-learning**.

## Built with
![Static Badge](https://img.shields.io/badge/Unity%202022.3.11f1-616161?style=for-the-badge&logo=unity&labelColor=black&link=https%3A%2F%2Funity.com%2F)
![Static Badge](https://img.shields.io/badge/Oculus%20HUb%20-%20%231C1E20?style=for-the-badge&logo=Oculus&labelColor=black)
![Static Badge](https://img.shields.io/badge/Meta%20Quest%20Developer%20Hub%20-%23616161?style=for-the-badge&logo=Meta&labelColor=black)
![Static Badge](https://img.shields.io/badge/Python%203.10.2-3776AB?style=for-the-badge&logo=python&logoColor=yellow&labelColor=black&link=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-3102%2F)
![Static Badge](https://img.shields.io/badge/numpy%202.1.4%20-%20%23013243?style=for-the-badge&logo=numpy&labelColor=black)
![Static Badge](https://img.shields.io/badge/pandas%201.26.2%20-%20%23150458?style=for-the-badge&logo=numpy&labelColor=black)
![Static Badge](https://img.shields.io/badge/scikit-learn%201.3.2%20-%20%23F7931E?style=for-the-badge&logo=scikit-learn&labelColor=black)
![Static Badge](https://img.shields.io/badge/Matplotlib%203.8.2-%20%23FFB71B?style=for-the-badge&logo=python&logoColor=%23FFB71B&labelColor=black)
![Static Badge](https://img.shields.io/badge/PyTorch%202.1.1%20-%20%23EE4C2C?style=for-the-badge&logo=PyTorch&labelColor=black)
![Static Badge](https://img.shields.io/badge/openaigym%200.26.2%20-%20%230081A5?style=for-the-badge&logo=openaigym&labelColor=black)

## Getting Started

##  ![Static Badge](https://img.shields.io/badge/%20-%20black?style=plastic&logo=Unity&logoColor=white&labelColor=black) Import the Unity Project ##
1. **Open Unity and create a new project:**
    - Launch Unity and select "New Project".

2. **Go to "Assets" and select "Import Package" > "Custom Package":**
    - In the Unity window, go to "Assets" in the menu bar.
    - Select "Import Package" and then choose "Custom Package".

3. **Choose the package to import:**
    - Navigate through your file system and select the Unity package (`MarcoSmilesOculusIntegration.unitypackage`) you downloaded from repo.

4. **Select the assets to import:**
    -  You  will see a list of all the assets in the package.
    - Ensure you selected all the assets.

5. **Click "Import".**



At this point, your project has been successfully imported, but there are additional steps required to complete the entire process.

#### From "Package Manager" download and install: ####

    - XR Plugin Managment (v 4.4.0)
    - XR Core Utilities (v 2.2.3)
    - XR Hands (v 1.3.0)
    - XR Interaction Toolkit (v 2.5.2)

##  ![Static Badge](https://img.shields.io/badge/%20-%203776AB?style=plastic&logo=python&logoColor=white&labelColor=%233776ab&color=%233776ab) Install Python Libraries ##
The libraries being installed henceforth are exclusively intended for deployment in the reinforcement training phase, 
given that Unity lacks native Python support. As of now, the wrapper enabling the 
utilization of trained agents within Unity remains unimplemented. Consequently, 
these libraries are deployed for research endeavors, aimed at discerning 
the optimal solution for MarcoSmiles.

`pip install numpy==2.1.4 pandas==1.26.2 
scikit-learn==1.3.2 matplotlib==3.8.2 torch==2.1.1 gym==0.26.2`

## Feature extraction ##
The information concerning hand positions is acquired through the hand tracking 
subsystem of the Meta Quest 3. Once acquired, it is processed using the XRHands 
package, which enables investigation into the position of each hand joint.
<p align="center"><img src="./README_IMAGES/XRHands.png" width="50%" height="50%">

In the next stage of our methodology, a fundamental calculation is performed, namely the
determination of the distance of each joint from the palm, with the exception of the wrist.
This calculation results in a set of data that intricately delineates the characteristics
of a given hand configuration.

      THUMB
         |-ThumbMetacarpalDistance
         |-ThumbProximalDistance
         |-ThumbIntermediateDistance
         |-ThumbDistalDistance
         |-ThumbTipDistance
      INDEX
         |-IndexMetacarpalDistance
         |-IndexProximalDistance
         |-IndexIntermediateDistance
         |-IndexDistalDistance
         |-IndexTipDistance
      MIDDLE
         |-MiddleMetacarpalDistance
         |-MiddleProximalDistance
         |-MiddleIntermediateDistance
         |-MiddleDistalDistance
         |-MiddleTipDistance
      RING
         |-RingMetacarpalDistance
         |-RingProximalDistance
         |-RingIntermediateDistance
         |-RingDistalDistance
         |-RingTipDistance
      LITTLE
         |-LittleMetacarpalDistance
         |-LittleProximalDistance
         |-LittleIntermediateDistance
         |-LittleDistalDistance
         |-LittleTipDistance

<p align="center"><img src="./README_IMAGES/HandSkeleton.png" width="100%" height="100%"><img src="./README_IMAGES/HandFeatures.png" width="100%" height="100%">

## Unity Scenes ##
### Training scene ###
The training scene facilitates the correlation of specific musical notes with corresponding hand positions. 
Within the scene, a panel is provided for the selection of the note to be trained. Once selected, 
a countdown initiates, during which the desired hand position must be assumed. Upon completion, 
the features pertaining to the recorded note are then stored.
After registering a note, it will be displayed in green on the virtual keyboard.

<p align="center"><img src="./README_IMAGES/TrainScene.png" width="100%" height="100%">











