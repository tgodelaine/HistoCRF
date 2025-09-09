import torch

device = 'cuda'


def unitary_potential_from_softmax(args, probabilities):
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(probabilities) 
    unitary_potential = -(torch.log(probabilities, device=device)) 
    return unitary_potential.T   