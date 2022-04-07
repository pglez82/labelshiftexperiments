import sys
sys.path.append('/media/nas/pgonzalez/DLquantification')
from histnet.histnet import HistNet
from histnet.featureextraction.fullyconnected import FCFeatureExtractionModule
from histnet.utils.utils import TestBagGenerator
from abstention.label_shift import PriorShiftAdapterFunc 
from torch.utils.data import TensorDataset
import torch

class HistNetAdapter():
    def __init__(self, model_path, random_seed=2032):
        device = torch.device('cpu')
        fe = FCFeatureExtractionModule(input_size=784, output_size=128, hidden_sizes=[256],dropout=0, activation="relu",flatten=True)
        self.histnet = HistNet(train_epochs = 0, test_epochs=1, start_lr=0,end_lr=0,random_seed=random_seed,
        n_bags=500,batch_size=100,linear_sizes=[512],bag_size=2000,n_bins=32, feature_extraction_module=fe, n_classes=10,bag_generator=None,
        save_model_path=model_path, device=device,verbose=0)

    def __call__(self, X, train_prevalence, test_prevalences):
        calibrator_func = lambda x: x
        self.histnet.bag_size_test = X.shape[0]
        return PriorShiftAdapterFunc(
                    multipliers=(self.histnet.predict(X)/train_prevalence),
                    calibrator_func=calibrator_func)