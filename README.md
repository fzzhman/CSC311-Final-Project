# CSC311-Final-Project

Fizzah: Currently working on Part A Q2 (log likelihood)

KNN Writeup:
  State the underlying assumption on itembased collaborative filtering: 
  Underlying Assumption:  The core underlying assumption is that if Question A is similar to Question B, then a student will score similarly on both Questions.
  
Q1 KNN Part (e) 
  Two potential drawbacks of knn for this task: 
  
    We can safely assume that there is high correlation between both question difficulty and student ability on whether or not the question was answered correctly. But, feature importance is not possible for the KNN algorithm (there is not an easy way which is defined to compute the features which are responsible for the classification), so it will not be able to make accurate inferences based on these two parameters. In the algorithm used in this question, either one of the two parameters (user ability or question difficulty) is focused on, so it has lower validation and test accuracy scores than other algoritms in Part A of this project.
    KNN runs slowly.
     
    

2 part (b): Hyperparameters selected:
  Theta: random values
  Beta: random values
  iterations: 20
  lr : 0.009 #this resulted in marginally higher accuracy for both training and validation sets.

2 part (d):  Comment on the shape of the curves and briefly describe what these curves represent.
  As value of theta increases, probability of student getting the answer to question [i] correct increases. 
