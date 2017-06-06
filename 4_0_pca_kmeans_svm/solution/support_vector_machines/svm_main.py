import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import preprocessing

def visualize_2D(features, labels, clf_a, clf_b, clf_c):
    h = 0.02
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    for i, clf in enumerate((clf_a, clf_b, clf_c)):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        if clf.probability == False:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            Z = Z[:,1].reshape(xx.shape)
        plt.subplot(2, 2, i + 1)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.coolwarm)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('SVMs')
    plt.show()
    
# Load features and labels from file.
# WARNING: These are different from the results of the previous exercises! Use the provided file!
features = np.genfromtxt(open("features_svm.txt"))
n_samples, n_features = np.shape(features)
labels = np.genfromtxt(open("labels_svm.txt"))
features_evaluation = np.genfromtxt(open("features_evaluation.txt"))
visualization = True

# TODO: Normalize your input feature data (the evaluation data is already normalized).
features = preprocessing.scale(features)

# TODO: Train 3 different classifiers as specified in the exercise sheet.
classifier_linear = svm.SVC(C=1, kernel='linear')
classifier_linear.fit(features, labels)

classifier_g_perfect = svm.SVC(C=1000000, kernel='rbf')
classifier_g_perfect.fit(features, labels)

classifier_g_slack = svm.SVC(C=1, kernel='rbf')
classifier_g_slack.fit(features, labels)

# TODO (optional): Train 3 different classifiers only on first 2 dimensions for visualization.
if visualization:
    classifier_linear_viz = svm.SVC(C=1, kernel='linear')
    classifier_linear_viz.fit(features[:,[0,1]], labels)

    classifier_g_perfect_viz = svm.SVC(C=1000000, kernel='rbf')
    classifier_g_perfect_viz.fit(features[:,[0,1]], labels)

    classifier_g_slack_viz = svm.SVC(C=1, kernel='rbf')
    classifier_g_slack_viz.fit(features[:,[0,1]], labels)

    visualize_2D(features, labels, classifier_linear_viz, classifier_g_perfect_viz, classifier_g_slack_viz)

# TODO: classify evaluation data and store classifications to file.
Z_linear = classifier_linear.predict(features_evaluation)
Z_g_perfect = classifier_g_perfect.predict(features_evaluation)
Z_g_slack = classifier_g_slack.predict(features_evaluation)

# Save probability results to file. 
# Z_linear, Z_g_perfect and Z_g_slack are of the form N x 1 dimensions,
#  with number of features N. 
np.savetxt('results_svm_Z_linear.txt', Z_linear)
np.savetxt('results_svm_Z_g_perfect.txt', Z_g_perfect)
np.savetxt('results_svm_Z_g_slack.txt', Z_g_slack)

# TODO: Train the same 3 classifiers as specified in the exercise sheet with additional probability estimates.
classifier_linear = svm.SVC(C=1, kernel='linear', probability=True)
classifier_linear.fit(features, labels)

classifier_g_perfect = svm.SVC(C=1000000, kernel='rbf', probability=True)
classifier_g_perfect.fit(features, labels)

classifier_g_slack = svm.SVC(C=1, kernel='rbf', probability=True)
classifier_g_slack.fit(features, labels)

# TODO (optional): Train 3 different classifiers with probability estimates only on first 2 dimensions for visualization.
if visualization:
    classifier_linear_viz = svm.SVC(C=1, kernel='linear', probability=True)
    classifier_linear_viz.fit(features[:, [0,1]], labels)

    classifier_g_perfect_viz = svm.SVC(C=1000000, kernel='rbf', probability=True)
    classifier_g_perfect_viz.fit(features[:, [0,1]], labels)

    classifier_g_slack_viz = svm.SVC(C=1, kernel='rbf', probability=True)
    classifier_g_slack_viz.fit(features[:, [0,1]], labels)

    visualize_2D(features, labels, classifier_linear_viz, classifier_g_perfect_viz, classifier_g_slack_viz)

# TODO: classify newly loaded features and store classification probabilities to file.
P_linear = classifier_linear.predict_proba(features_evaluation)
P_g_perfect = classifier_g_perfect.predict_proba(features_evaluation)
P_g_slack = classifier_g_slack.predict_proba(features_evaluation)

# Save probability results to file. 
# P_linear, P_g_perfect and P_g_slack are of the form N x 2 dimensions,
#  with number of features N and classification probabilities for the two classes. 
np.savetxt('results_svm_P_linear.txt', P_linear)
np.savetxt('results_svm_P_g_perfect.txt', P_g_perfect)
np.savetxt('results_svm_P_g_slack.txt', P_g_slack)