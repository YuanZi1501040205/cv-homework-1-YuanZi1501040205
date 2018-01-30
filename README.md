# Computer Vision
Homework #1

Due: Tu 02/13/18 11:59 PM

1. Kmeans Segmentation (Grey Scale):
(5 Pts.) Write code to perform segmentation of a grey scale image using KMeans clustering

  - Starter code available in directory Segmentation/
  - Segmentation/KMeans.py: One is required to edit the functions "segmentation_grey". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns the segmented image
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hw1.py (there is no need to edit this file)
  - Usage: ./cv_hw1 -i image -k clusters -m grey
            python cv_hw1.py -i image -k clusters -m grey
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hw1.py automatically does this)
  
-------------
2. Kmeans Segmentation (Color):
(5 Pts.) Write code to perform segmentation on a color image using KMeans clustering

  - Starter code available in directory Segmentation/
  - Segmentation/KMeans.py: One is required to edit the functions "segmentation_rgb". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns the segmented image
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hw1.py (there is no need to edit this file)
  - Usage: ./cv_hw1 -i image -k clusters -m rgb
            python cv_hw1.py -i image -k clusters -m rgb
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hw1.py automatically does this)

-------------
3. (2 Pts.) Describe your method and report you findings in your report for each problem of the assignemnt.
  - In your report decribe the following points
    a. What are the parameters that influence your algorithm? Explain their effect
    b. Does your segementation code always generate the same segments for a given k. Explain?
    c. What is the objective of this implementations? Which set of features perform better (color vs greyscale)? Explain
-------------

Four images are provided for testing: cirle, fruits, kenny and lemons
PS. Files not to be changed: requirements.txt and .circleci directory 
If you do not like the structure, you are welcome to change the over all code, under two stipulations:

1. the homework has to run using command

  python cv_hw1.py -i image -k clusters -m rgb
  and
  python cv_hw1.py -i image -k clusters -m grey
 

  
2. Any output file or image should be written to output/ folder

The TA will only be able to see your results if these two conditions are met

1. Kmeans grey     - 5 Pts.
2. Kmenas color    - 5 Pts.
3. Report          - 5 Pts

    Total          - 15 Pts.

----------------------
