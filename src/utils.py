import os
import pickle
def get_model_info(model_name, suffix= "", folder = '../results/'):
    # get accuracy
    acc_str = "ACCURACY_" + model_name + suffix +  ".pkl"
    acc_path = os.path.join(folder, acc_str)
    with open(acc_path, 'rb') as f:
        acc = pickle.load(f)
    # get class_accuracy
    class_acc_str = "CLASS_ACCURACY_" + model_name + suffix + ".pkl"
    class_acc_path = os.path.join(folder, class_acc_str)
    with open(class_acc_path, 'rb') as f:
        class_acc = pickle.load(f)
    # get confusion matrix
    conf_str = "CONF_MAT_" + model_name + suffix + ".pkl"
    conf_path = os.path.join(folder, conf_str)
    with open(conf_path, 'rb') as f:
        conf = pickle.load(f)
    # get history
    hist_str = "HISTORY_" + model_name + suffix + ".pkl"
    hist_path = os.path.join(folder, hist_str)
    with open(hist_path, 'rb') as f:
        hist = pickle.load(f)
    dict_ = {
        "acc": acc, 
        "class_acc": class_acc,
        "conf_matrix": conf, 
        "hist": hist}
    return dict_
