#Write the training procedure for the Model
sys.path.append(".")
from CustomLoss import CustomLoss
from DataCollect import assemble_dataset
from PINN import PINN, init_xavier

#Train File
tau_max = 1
nu_max = 1
n_collocation = 10000
n_boundary = 10000
seed = 5
network_depth = 7
network_width = 5
lambda_loss = 1
interior, boundary_1, boundary_2 = assemble_dataset(tau_max, nu_max, n_collocation, n_boundary, seed)
pinn = PINN(network_depth, network_width)
MAX_EPOCHS = 10000
LRATE = 3e-4

#Initialize Neural Network
init_xavier(pinn)

#Use Adam for training(change to BFGS, normalization?)
optimizer = torch.optim.Adam(pinn.parameters(), lr=LRATE)

loss_history_u = []
loss_history_f = []
loss_history = []
c_loss = CustomLoss()

for epoch in range(MAX_EPOCHS):
    #full batch
    loss = c_loss(pinn, interior, boundary_1, boundary_2, lambda_loss)
    loss_history.append([epoch, loss])

    #optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print("Epoch: {}, MSE: {:.4f}".format((epoch+1), loss))
