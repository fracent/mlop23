"""This file was created x mlop dtu course 2023,
francesco centomo developed this code with help from
chat gpt4 and mlop material"""


from model import ClassifierWithDropout, train_and_validate
from data import load_data

if __name__ == "__main__":
    trainloader, testloader = load_data()
    model = ClassifierWithDropout()
    train_and_validate(model, trainloader, testloader)
