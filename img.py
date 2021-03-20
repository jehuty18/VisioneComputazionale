import os
from random import seed
import src.lib.network as net
import src.lib.image_utils as imgt
import src.lib.enum_utils as eut

def main():
    #BASE-TEST with the original dataset for images
    train_dir_base_path = "./src/resources/imgs/train"
    test_dir_base_path = "./src/resources/imgs/test"
    #Feature extraction for all images under 'train_dir_base_path' directory
    dataset = imgt.feature_extraction_from_dir(train_dir_base_path)
    print("[BASE-TEST] after feature matrix created")
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
    print("\n[BASE-TEST] after test feature matrix created")
    for row in testset:
        prediction = net.predict(network, row, eut.TraningFunctions.SIGMOID)
        if row[-1] is None:
            print('Blind evaluation, Got=%d' % (prediction))
        else:	
            print('Expected=%d, Got=%d' % (row[-1], prediction))

    #DOCT-TEST with doctored images doctored again (hinstogram alterated)
    train_dir_base_path = "./src/resources/supervised_doct/train"
    test_dir_base_path = "./src/resources/supervised_doct/test"
    test_unsupervised_dir_base_path = "./src/resources/unsupervised/test"
    #Feature extraction for all images under 'train_dir_base_path' directory, this time using a subset of doctered images doctored twice
    dataset = imgt.feature_extraction_from_dir(train_dir_base_path)
    print("\n[DOCT-TEST] after feature matrix created")
    #setting parameters for network definition
    n_inputs = len(dataset[0]) - 1
    n_hidden = int(len(dataset) / 2) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    l_rate = 0.5
    epochs = 200
    seed(1)
    print("n_inputs: %f\tn_hidden: %f\tn_outputs: %f" % (n_inputs, n_hidden, n_outputs))
    #Network initialization + network training (second training set)
    network = net.initialize_network(n_inputs, n_hidden, n_outputs)
    net.train_network(network, dataset, l_rate, epochs, n_outputs, eut.TraningFunctions.SIGMOID)
    print("net trained")
    #feature extraction for test set + network usage for prediction
    testset = imgt.feature_extraction_from_dir(test_dir_base_path)
    print("\n[DOCT-TEST] after test feature matrix created")
    for row in testset:
        prediction = net.predict(network, row, eut.TraningFunctions.SIGMOID)
        if row[-1] is None:
            print('Blind evaluation, Got=%d' % (prediction))
        else:	
            print('Expected=%d, Got=%d' % (row[-1], prediction))
    #feature extraction for test set + network usage for prediction in unsupervised scenario
    testset_uns = imgt.feature_extraction_from_dir_unsupervised(test_unsupervised_dir_base_path)    
    print("[DOCT-TEST] after testset_uns feature matrix created")
    for row in testset_uns:
        prediction = net.predict(network, row, eut.TraningFunctions.SIGMOID)
        if row[-1] is None:
            print('Blind evaluation, Got=%d' % (prediction))
        else:	
            print('Expected=%d, Got=%d' % (row[-1], prediction))

if __name__ == "__main__":
    main()