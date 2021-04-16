import os
import tensorflow as tf


def learningRateCallback(initial_learning_rate=0.001,
                         decay_steps=100000,
                         decay_rate=0.96,
                         staircase=True):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase)
    return lr_schedule


#######################################################################################################################
def tensorboardCallback(save_dir, histogram_freq=0, embedding_freq=0, write_graph=False):
    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=save_dir,
                                                     histogram_freq=0,  # How often to log histogram visualizations.
                                                     embeddings_freq=0,  # How often to log embedding visualizations.
                                                     write_graph=False, )  # visualize the graph in TensorBoard. Log
    return tensorboard_cbk


def saveModelOnMetric(save_dir, metric='val_loss', mode='min', save_freq='epoch', save_best_only=True,
                      save_weights_only=True):
    if save_weights_only:
        checkpoint_path = os.path.join(save_dir, 'model_weights/best_model_on_{}_weights.h5'.format(metric))
    else:
        checkpoint_path = os.path.join(save_dir, 'model_weights/best_model_on_{}.h5'.format(metric))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             monitor=metric,
                                                             save_freq=save_freq,
                                                             save_best_only=save_best_only,
                                                             save_weights_only=save_weights_only,
                                                             mode=mode)
    return checkpoint_callback


# #######################################################################################################################
# class metricStatistics(tf.keras.callbacks.Callback):
#
#     def __init__(self, train_data, val_data, save_dir, classes, save_freq=10):
#         self.train_data = train_data
#         self.val_data = val_data
#         self.save_dir = save_dir
#         self.save_freq = save_freq
#         self.classes = classes
#         # metrics
#         self.binary_accuracy, self.val_binary_accuracy = [], []
#         self.fp, self.val_fp = [], []
#         self.tp, self.val_tp = [], []
#         self.loss, self.val_loss = [], []
#         self.tn, self.val_tn = [], []
#         self.fn, self.val_fn = [], []
#         self.recall, self.val_recall = [], []
#         self.f1, self.val_f1 = [], []
#         self.precision, self.val_precision = [], []
#         self.auc, self.val_auc = [], []
#         self.specificity, self.val_specificity = [], []
#         # we can add more metrics here and in the function get_metrics()
#
#     # Function for calculating model.predict() on a tf Dataset
#     def predict_on_data(self, data, name='train'):
#         if name == 'validation':
#             filenames = np.array(list(data.flat_map(
#                 lambda image, label, filename: tf.data.Dataset.from_tensor_slices(filename)).as_numpy_iterator()))
#             data = data.map(lambda image, label, filename: (image, label))
#
#         ##########################################
#         # Get true_labels, predicted_values
#         ##########################################
#         true_labels = np.array(
#             list(data.flat_map(lambda image, label: tf.data.Dataset.from_tensor_slices(label)).as_numpy_iterator()))
#         predicted_values = self.model.predict(data)
#         predicted_values = np.reshape(predicted_values, -1)
#
#         ##########################################
#         # return depending on the names
#         # train: return true_labels, predicted_values
#         # validation: return true_labels, predicted_values, filenames
#         ##########################################
#         if name == 'validation':
#             return true_labels, predicted_values, filenames
#         else:
#             return true_labels, predicted_values
#
#     # Function used for plotting ROC, metrics
#     def plots(self, epoch, logs):
#
#         train_labels, train_values = self.predict_on_data(self.train_data, 'train')
#         val_labels, val_values, val_filenames = self.predict_on_data(self.val_data, 'validation')
#
#         ##########################################
#         # plot roc curves
#         ##########################################
#
#         sensitivity_values = [0.975, 0.95, 0.90, 0.85]
#         plt.figure(figsize=(10, 10))
#         auct = plot_roc(name='Train ROC',
#                         y_true=train_labels,
#                         y_pred=train_values,
#                         color='b',
#                         linestyle='--',
#                         linewidth=2)
#         aucv = plot_roc_detailed(name='Validation ROC',
#                                  y_true=val_labels,
#                                  y_pred=val_values,
#                                  color='b',
#                                  sensitivity_values=sensitivity_values)
#         plt.title(
#             'Epoch: {0:.2f}, Train AUC (all/batch): {1:.2f}/{2:.2f}, val. auc (all/batch): {3:.2f}/{4:.2f}'.format(
#                 epoch, logs['AUC'], auct, logs['val_AUC'], aucv))
#         plt.legend(loc='lower right')
#         plt.savefig(os.path.join(self.save_dir, 'metrics/detailed_roc_epoch_' + str(epoch) + '.png'))
#         plt.close()
#
#         ############################################
#
#         ############################################
#         # plot metrics
#         ############################################
#         try:
#             # Training loss
#             plt.figure(figsize=(10, 10))
#             plt.subplot(6, 1, 1)
#             plt.plot(self.loss, 'k', label='Training Loss')
#             plt.plot(self.val_loss, 'k--', label='Validation Loss')
#             plt.yscale('log')
#             plt.title('Loss')
#             plt.legend()
#
#             # Recall
#             plt.subplot(6, 1, 2)
#             plt.plot(self.recall, 'r', label='Recall')
#             plt.plot(self.val_recall, 'r--')
#             plt.title('Recall (Sensitivity)')
#             plt.legend()
#
#             # Specificity
#             plt.subplot(6, 1, 3)
#             plt.plot(self.specificity, 'r', label='Specificity')
#             plt.plot(self.val_specificity, 'r--')
#             plt.title('Specificity')
#             plt.legend()
#
#             # precision per class
#             plt.subplot(6, 1, 4)
#             plt.plot(self.precision, 'r', label='Precision')
#             plt.plot(self.val_precision, 'r--')
#             plt.title('Precision')
#             plt.legend()
#
#             # F1 per class
#             plt.subplot(6, 1, 5)
#             plt.plot(self.f1, 'r', label='F1')
#             plt.plot(self.val_f1, 'r--')
#             plt.title('F1')
#             plt.legend()
#
#             # AUC
#             plt.subplot(6, 1, 6)
#             plt.plot(self.auc, 'r', label='AUC')
#             plt.plot(self.val_auc, 'r--')
#             plt.title('AUC')
#             plt.legend()
#
#             plt.tight_layout()
#
#             plt.savefig(os.path.join(self.save_dir, 'metrics/metrics_epoch_{}.png'.format(epoch)))
#             plt.close()
#         except ValueError as e:
#             print(e)
#
#         return train_labels, train_values, val_labels, val_values, val_filenames
#
#     def get_best_F1_threshold_PR_curve(self, y_true, y_pred):
#         '''
#         Calculate the best F1 and corresponding Threshold using PR curve for the given labels and true predictions
#         '''
#         precision, recall, threshold = sklearn.metrics.precision_recall_curve(y_true, y_pred)
#
#         # Calculate minimum distance from the point (1, 1)
#         dist = np.sqrt(np.square(recall - 1) + np.square(precision - 1))
#         min_dist_index = np.argmin(dist)
#         best_precision = precision[min_dist_index]
#         best_recall = recall[min_dist_index]
#         best_threshold = threshold[min_dist_index]
#         best_f1 = 2 * best_precision * best_recall / (best_recall + best_precision)
#
#         return best_f1, best_threshold
#
#     def get_best_F1_threshold_ROC_curve(self, y_true, y_pred):
#         '''
#         Calculate the best F1 and corresponding Threshold using ROC curve for the given labels and true predictions
#         '''
#         fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true, y_pred)
#
#         # Calculate minimum distance from the point (0, 1)
#         dist = np.sqrt(np.square(fpr - 0) + np.square(tpr - 1))
#         min_dist_index = np.argmin(dist)
#         best_threshold = threshold[min_dist_index]
#
#         # Get the corresponding confusion_matrix using the calculated threshold
#         tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, np.where(y_pred > best_threshold, 1, 0)).ravel()
#         best_precision = tp / (tp + fp)
#         best_recall = tp / (tp + fn)
#         best_f1 = 2 * best_precision * best_recall / (best_precision + best_recall)
#
#         return best_f1, best_threshold
#
#     # Function for saving val_predictions with filenames
#     def save_validation_predictions(self, epoch, labels, values, filenames, name=None):
#         if name is None:
#             csv_filename = os.path.join(self.save_dir, 'metrics/val_predictions_at_epoch_{}.csv'.format(epoch))
#         else:
#             csv_filename = os.path.join(self.save_dir, name)
#
#         with open(csv_filename, 'a') as fp:
#             wr = csv.writer(fp, dialect='excel')
#             wr.writerow(['Image name', 'Model Prediction', 'True Label'])
#             for i in range(len(labels)):
#                 try:
#                     name = filenames[i].decode('utf-8').split('/')[-1]
#                 except AttributeError as e:
#                     name = filenames[i].split('/')[-1]
#                 wr.writerow([name, round(values[i], 4), labels[i]])
#             fp.close()
#
#     # # Function for Confusion matrix
#     # def log_confusion_matrix(self, epoch, logs, train_labels, train_values):
#     #     filewriter_cm = tf.summary.create_file_writer(self.save_dir + '/cm')
#     #     train_predicted_labels = np.argmax(train_values, axis=0)
#     #     cm = sklearn.metrics.confusion_matrix(train_labels, train_predicted_labels)
#     #     figure = plot_confusion_matrix(cm, class_names=self.classes)
#     #     cm_image = plot_to_image(figure)
#     #     with filewriter_cm.as_default():
#     #         tf.summary.image("Conf Matrix (on batch of TRAIN data)", cm_image, step=epoch)
#
#     # Function that is called at the end of every epoch in the callback
#     def on_epoch_end(self, epoch, logs=None):
#
#         self.binary_accuracy.append(logs['Binary Accuracy'])
#         self.val_binary_accuracy.append(logs['val_Binary Accuracy'])
#
#         self.loss.append(logs['loss'])
#         self.val_loss.append(logs['val_loss'])
#
#         self.fp.append(logs['False Positives'])
#         self.val_fp.append(logs['val_False Positives'])
#
#         self.tp.append(logs['True Positives'])
#         self.val_tp.append(logs['val_True Positives'])
#
#         self.tn.append(logs['True Negatives'])
#         self.val_tn.append(logs['val_True Negatives'])
#
#         self.fn.append(logs['False Negatives'])
#         self.val_fn.append(logs['val_False Negatives'])
#
#         self.precision.append(logs['Precision'])
#         self.val_precision.append(logs['val_Precision'])
#
#         self.recall.append(logs['Recall'])
#         self.val_recall.append(logs['val_Recall'])
#
#         self.auc.append(logs['AUC'])
#         self.val_auc.append(logs['val_AUC'])
#
#         # F1 score
#         F1 = 2 * np.multiply(self.recall[-1], self.precision[-1]) / (
#                 np.array(self.recall[-1]) + np.array(self.precision[-1]))
#         val_F1 = 2 * np.multiply(self.val_recall[-1], self.val_precision[-1]) / (
#                 np.array(self.val_recall[-1]) + np.array(self.val_precision[-1]))
#         self.f1.append(F1)
#         self.val_f1.append(val_F1)
#
#         # Specificity
#         specificity_value = np.array(self.tn[-1]) / (np.array(self.tn[-1]) + np.array(self.fp)[-1])
#         val_specificity_value = np.array(self.val_tn[-1]) / (np.array(self.val_tn[-1]) + np.array(self.val_fp)[-1])
#         self.specificity.append(specificity_value)
#         self.val_specificity.append(val_specificity_value)
#
#         if epoch != 0 and epoch % self.save_freq == 0:
#             train_labels, train_values, val_labels, val_values, val_filenames = self.plots(epoch, logs)
#
#             #####################################
#             # save validation predictions in csv
#             #####################################
#             self.save_validation_predictions(epoch, val_labels, val_values, val_filenames)
#
#             #####################################
#             # Confusion matrix
#             #####################################
#             # self.log_confusion_matrix(epoch, logs, train_labels, train_values)
#
#             #####################################
#             # save model weights every 40 epochs
#             #####################################
#             # if epoch % 4 == 0:
#             # saving model for every epoch
#             self.model.save_weights(os.path.join(self.save_dir, 'model_weights/weights_epoch_{}.h5'.format(epoch)))
#             # self.model.save(os.path.join(self.save_dir, 'model_weights/model_epoch_{}.h5'.format(epoch)))