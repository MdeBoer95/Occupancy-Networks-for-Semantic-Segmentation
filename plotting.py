import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Keys in logfile: iteration, points_accuracy, tp, fp, tn, fn, precision, recall

log_file = "metric_log"
pd_stats = pd.read_csv(log_file)
print(pd_stats)
epoch_acc_plotpoints = pd_stats[['iteration', 'points_accuracy']].to_numpy()
epoch_pre_plotpoints = pd_stats[['iteration', 'precision']].to_numpy()
epoch_rec_plotpoints = pd_stats[['iteration', 'recall']].to_numpy()
epoch_iou_plotpoints = pd_stats[['iteration', 'iou_complete']].to_numpy()
plt.interactive(False)
plt.figure()
line1, = plt.plot(epoch_acc_plotpoints[:, 0], epoch_acc_plotpoints[:, 1], color='green')
line2, = plt.plot(epoch_pre_plotpoints[:, 0], epoch_pre_plotpoints[:, 1], color='cyan')
line3, = plt.plot(epoch_rec_plotpoints[:, 0], epoch_rec_plotpoints[:, 1], color='red')
line4, = plt.plot(epoch_iou_plotpoints[:, 0], epoch_iou_plotpoints[:, 1], color='blue')
line1.set_label('Accuracy')
line2.set_label('Precision')
line3.set_label('Recall')
line4.set_label('IoU')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('plot0_4.png')
print("done")