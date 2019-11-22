Execute: python HW2.py [train_directory] [test_directory]

example: python HW3.py "netflix\TrainingRatings.txt" "netflix\TestingRatings.txt"

(This will load the data from the Training and Testing files. The program will pick a user from the testing file and calculate the weight matrix for just that user. Then it will predict the rating for all the movie for that user in the test file. Finally the program will calculate the Mean Absolute Error and Root Mean Squared Error for all the examples in the test file and display their values.)

Output:

Total users: 27555
Iteration: 1 UserID: 7 Time Elapsed: 19.232882022857666
.
.
.
Iteration: 27555 UserID: 2649285 Time Elapsed: 36435.232882022857666
Mean Absolute Error: 0.6949234976496896 & Root Mean Squared Error: 0.8844601266219813
Total Time Taken: 36437.015588998794555664
