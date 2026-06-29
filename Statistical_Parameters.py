from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

true_labels = [1,1,0,0]
pred_labels = [0,1,0,0]

# Calculate scores
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Print scores
print(f"Precision (macro): {precision:.2f}")
print(f"Recall (macro):    {recall:.2f}")
print(f"F1-score (macro):  {f1:.2f}")
print("Confusion Matrix:\n", conf_matrix)
