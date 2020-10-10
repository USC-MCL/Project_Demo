import InterpretableCNN
import data
import sys

def main():
	# read data
    dataset = sys.argv[1]
    train_images, train_labels, test_images, test_labels, class_list = data.import_data(dataset, "0-9")
    print('Training image size:', train_images.shape)
    print('Testing_image size:', test_images.shape)
    myCNN=InterpretableCNN.InterpretableCNN(train_images, train_labels, test_images, test_labels, class_list)
    myCNN.getKernel()
    myCNN.getFeature()
    myCNN.getWeight()


if __name__ == '__main__':
	main()