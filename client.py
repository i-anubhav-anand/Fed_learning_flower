import numpy as np

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import flwr as fl

from typing import Tuple, Dict
import covid




if torch.cuda.is_available():  #PyTorch has device object to load the data into the either of two hardware [CPU or CUDA(GPU)]
    DEVICE=torch.device("cuda:0")
    print("Training on GPU")
else:
    DEVICE = torch.device("cpu")
    print("Training on CPU")



DATA_PATH = 'dataset/dataset'



class Covid_Fed_Client(fl.client.NumPyClient):
    """Flower client implementing covid-19 image classification using
    PyTorch."""

    def __init__(
        self,
        model: covid.Net(),
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        data_sizes: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.data_sizes = data_sizes
        #return the model weight as a list of NumPy ndarrays
    def get_parameters(self) -> List[np.ndarray]: 
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # set the local model weights
        # train the local model
        # receive the updated local model weights
        self.set_parameters(parameters)
        covid.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(), self.data_sizes["train"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = covid.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.data_sizes["train"], {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, from local directory."""

    # Load model and data
    model = covid.Net()
    model.to(DEVICE)
    trainloader, testloader, num_examples = covid.load_data(DATA_PATH,0.2)
    

    # created an instance of our class Covid_Fed_Client and add one line to actually run this client:
    client = Covid_Fed_Client(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client("34.219.109.134:8080", client)


if __name__ == "__main__":
    main()


