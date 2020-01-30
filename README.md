# Computer Vision
Homework #1

Due: Tu 02/09/20 11:59 PM

1. Kmeans Segmentation (Grey Scale):
(5 Pts.) Write code to perform segmentation of a grey scale image using KMeans clustering

  - Starter code available in directory Segmentation/
  - Segmentation/KMeans.py: You should edit the function "segmentation_grey". You are welcome to add more functions or add nested functions within the function
  - Make sure the function returns the segmented image
  - For this part of the assignment, please implement your own code for all computations. Do not use inbuilt functions from any packages including numpy, opencv or other libraries for clustering or segmentation
  - Test your solution for images provided and by specifying different number of clusters (k)
  - Describe your method and findings in your report, particularly the obtained result as it relates to the choice of k.  Reflect on both the quality of segmentation and the computational time.
  - This part of the assignment can be run using cv_hw1.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hw1 -i image -k clusters -m grey
  
            python cv_hw1.py -i image -k clusters -m grey
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hw1.py automatically does this)
  
-------------
2. Kmeans Segmentation (Color):
(5 Pts.) Write code to perform segmentation on a color image using KMeans clustering

  - Starter code available in directory Segmentation/
  - Segmentation/KMeans.py: You should edit the function "segmentation_rgb". You are welcome to add more functions or add nested functions within the function
  - Make sure the function returns the segmented image
  - For this part of the assignment, please implement your own code for all computations. Do not use inbuilt functions from any packages including numpy, opencv or other libraries for clustering or segmentation
  - Test your solution for images provided and by specifying different number of clusters (k)
  - Describe your method and findings in your report, particularly the obtained result as it relates to the choice of k.  Reflect on both the quality of segmentation and the computational time.
  - This part of the assignment can be run using cv_hw1.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hw1 -i image -k clusters -m rgb
  
            python cv_hw1.py -i image -k clusters -m rgb
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hw1.py automatically does this)

-------------
3. (5 Pts.) Describe your method and report you findings in your report for each of the implementations (greyscale and color).
  - Your report should accompany your code. 
  - Include a word/pdf file in the repository.
  - In your report, address the following:
  
    a. What are the parameters that influence your algorithm? Explain their effect?
    
    b. Does your segementation code always generate the same segments for a given k. Explain?
    
    c. What is the objective of this implementation? Which set of features perform better (color vs greyscale)? Explain?
    
    d. Is there any relationship between the choice of k and the time it takes to segment an image?  If so, explain.
    
    e. Is there a relationship between compute time and the number of image attributes (1d for grey scale vs. 3d for color)?  If so, explain.
    
-------------

Four images are provided for testing: cirle, fruits, kenny and lemons
OpenCV: You can use opencv functions to load, display images.

Make sure your final submission is running on circleci. 
The TA will use CircleCI output and your github code for grading. 
TA will not be able to grade if the code does not run on circle CI.

Common reasons for failure.

Do not use any 3rd party libraries or functions.
Do not display images in your final submission. 
Example, cv2.imshow(), cv2.waitkey(), cv2.NamedWindow will result in failure during circleci testing.

PS. Files not to be changed: requirements.txt and .circleci directory 
If you do not like the structure, you are welcome to change the over all code, under two stipulations:

1. the homework has to run using command

  python cv_hw1.py -i image -k clusters -m rgb
  and
  python cv_hw1.py -i image -k clusters -m grey
 

  
2. Any output file or image should be written to output/ folder

The TA will only be able to see your results if these two conditions are met

1. Kmeans grey     - 5 Pts.
2. Kmeans color    - 5 Pts.
3. Report          - 5 Pts

    Total          - 15 Pts.

----------------------
