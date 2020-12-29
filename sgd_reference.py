import torch
from torchvision import datasets, transforms

# Define neural network model
log_interval = 1200
torch.manual_seed(0)

neurons = [784, 512, 256, 10]
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(neurons[0], neurons[1], bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(neurons[1], neurons[2], bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(neurons[2], neurons[3], bias=False),
    torch.nn.Softmax(dim=1)
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = torch.nn.CrossEntropyLoss()

# Download and prepare MNIST dataset
image_cnt  = 60000
batch_size = 1
batch_cnt  = int(image_cnt / batch_size)

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
test_dataset  = datasets.MNIST('data', train=False, transform=transforms.ToTensor()) 
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

epoch_cnt = 1
for epoch in range(epoch_cnt):
    # Train network
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= batch_cnt: break

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), image_cnt,
                100. * batch_idx / batch_cnt, loss.data.item()))

    # Test network
    model.eval()

    correct = 0
    for data_idx, (data, target) in enumerate(test_loader):

        output  = model(data)
        pred    = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)

    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset), accuracy))
