import os
import cv2
import numpy as np


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        """
        :param preprocessors: List of image preprocessors
        """
        # store the image preprocessor
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        """ Used to load a list of images for pre-processing

        :param image_paths: List of image paths
        :param verbose: Parameter for printing information to console
        :return: Tuple of data and labels
        """
        data, labels = [], []

        for i, image_path in enumerate(image_paths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            #   /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]  # {class}

            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels.
            data.append(image)
            labels.append(label)

            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print('[INFO]: Processed {}/{}'.format(i + 1, len(image_paths)))

        #  return a tuple of the data and labels
        return (np.array(data), np.array(labels))
