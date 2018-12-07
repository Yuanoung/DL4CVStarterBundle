# import the necessary packages
import argparse
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier  # k-NN algorithm
# a helper utility to convert labels represented as strings to integer,
# where there is one unique integer per class label
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # create our training and testing splits.
# evaluate the performance of our classifier and print a nicely formatted table of results to our console
from sklearn.metrics import classification_report

from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
ap.add_argument('-n', '--neighbors', required=False, type=int, default=1,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', required=False, type=int, default=-1,  # concurrent jobs
                help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print('[INFO]: Images loading....')
image_paths = list(paths.list_images(args['dataset']))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)

# Reshape from (3000, 32, 32, 3), indicating there are 3,000 images in the dataset,
# each 32 ×32 pixels with 3 channels.
# 然而，为了应用k-NN算法，我们需要将我们的图像从3D表示“平坦化”为单个像素强度列表。
# 我们为了实现这一点，在第30行调用数据NumPy数组上的.reshape方法，将32×32×3图像展平为具有形状（3000,3072）的数组。
# 实际图像数据根本没有改变 - 图像简单地表示为3,000个条目的列表，每个条目为3,072-dim（32×32×3 = 3072）。
data = data.reshape((data.shape[0], 3072))

# Print information about memory consumption
print('[INFO]: Features Matrix: {:.1f}MB'.format(float(data.nbytes / 1024 * 1000.0)))

# Encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split data into training (75%) and testing (25%) data
# Therefore, we use the variables trainX and testX to refer to the
# training and testing examples, respectively. The variables trainY and testY are our training and
# testing labels.
# x: data
# y: label
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Train and evaluate the k-NN classifier on the raw pixel intensities
print('[INFO]: Classification starting....')
model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                             n_jobs=args['jobs'])
model.fit(train_x, train_y)
# le.classes_: label of string
print(classification_report(test_y, model.predict(test_x),
                            target_names=le.classes_))
