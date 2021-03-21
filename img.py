import os
import getopt, sys
from random import seed
import src.lib.network as net
import src.lib.image_utils as imgt
import src.lib.enum_utils as eut

short_options = "hr:t:u:"
long_options = ["help", "trainPath=", "testPath=", "unsupTestPath="]

def _parsing_args():
    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]
    
    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        print (str(err))
        sys.exit(2)
    
    for current_argument, current_value in arguments:
        if current_argument in ("-r", "--trainPath"):
            print (("Executing scripts using '%s' as trainingSet path") % (current_value))
            global train_dir_base_path
            train_dir_base_path = current_value
        elif current_argument in ("-h", "--help"):
            print ("Displaying help..\n\n\t-t --testPath(Required):\tpath for test images\n\t-r --trainPath(Required):\tpath for train images\n\t-u --unsupTestPath(Optional):\tpath for test images, unsupervised scenario\n")
        elif current_argument in ("-t", "--testPath"):
            print (("Executing scripts using '%s' as testSet path") % (current_value))
            global test_dir_base_path
            test_dir_base_path = current_value
        elif current_argument in ("-u", "--unsupTestPath"):
            print (("Executing scripts using '%s' as unsupervisedTestPath path") % (current_value))
            global test_unsupervised_dir_base_path
            test_unsupervised_dir_base_path = current_value

def main():

    _parsing_args()
    
    global test_dir_base_path
    global train_dir_base_path
    global test_unsupervised_dir_base_path

    #Assert-like block to check for variable definition (global directive does not define/initialize variables)
    try:
        train_dir_base_path
        test_dir_base_path
    except:
        print("TrainingSet path and/or TestSet path not valorized.\nExiting script...")
        exit(1)

    #Feature extraction for all images under 'train_dir_base_path' directory
    dataset = imgt.feature_extraction_from_dir(train_dir_base_path)
    print("[TEST] after feature matrix created")
    #setting parameters for network definition
    n_inputs = len(dataset[0]) - 1
    n_hidden = int(len(dataset) / 2) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    l_rate = 0.5
    epochs = 200
    seed(1)
    print("n_inputs: %f\tn_hidden: %f\tn_outputs: %f" % (n_inputs, n_hidden, n_outputs))
    #Network initialization + network training (first training set)
    network = net.initialize_network(n_inputs, n_hidden, n_outputs)
    net.train_network(network, dataset, l_rate, epochs, n_outputs, eut.TraningFunctions.SIGMOID)
    print("net trained")
    #feature extraction for test set + network usage for prediction
    testset = imgt.feature_extraction_from_dir(test_dir_base_path)
    print("\n[TEST] after test feature matrix created")
    for row in testset:
        prediction = net.predict(network, row, eut.TraningFunctions.SIGMOID)
        if row[-1] is None:
            print('Blind evaluation, Got=%d' % (prediction))
        else:	
            print('Expected=%d, Got=%d' % (row[-1], prediction))

    try:
        #feature extraction for test set + network usage for prediction in unsupervised scenario
        testset_uns = imgt.feature_extraction_from_dir_unsupervised(test_unsupervised_dir_base_path)    
        print("[TEST] after testset_uns feature matrix created")
        for row in testset_uns:
            prediction = net.predict(network, row, eut.TraningFunctions.SIGMOID)
            if row[-1] is None:
                print('Blind evaluation, Got=%d' % (prediction))
            else:	
                print('Expected=%d, Got=%d' % (row[-1], prediction))
    except:
        print("\nNo unsupervised test required")

if __name__ == "__main__":
    main()