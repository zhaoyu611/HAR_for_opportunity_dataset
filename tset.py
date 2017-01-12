import numpy as np
import matplotlib.pyplot as plt
a = [[476, 1, 20,0,0,0],
	 [20, 429, 21, 0, 0, 0],
	 [14, 8, 398, 0, 0, 0],
	 [0, 5, 0, 426, 33, 4],
	 [0, 0, 0, 62, 473, 0],
	 [0, 0, 0, 0, 0, 537]]
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
] 
confusion_matrix=np.array(a,dtype=np.float32)
sum_row_list=confusion_matrix.sum(1)
sum_col_list=confusion_matrix.sum(0)
for i in range(6):
	print confusion_matrix[i,i]/sum_row_list[i]
# print sum(confusion_matrix.max(1))/confusion_matrix.sum()


normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print ""
print "Confusion matrix (normalised to % of total test confusion_matrix):"
print normalised_confusion_matrix
print ("Note: training and testing confusion_matrix is not equally distributed amongst classes, "
       "so it is normal that more than a 6th of the confusion_matrix is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
# plt.title("Confusion matrix \n(normalised to % of total test confusion_matrix)")
plt.colorbar()
tick_marks = np.arange(6)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.show()