
#Custom Loss to minimize the residual on the interior and boudary data

class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, network, interior_data, boundary_data_1, boundary_data_2):


      optimizer = optim.Adam(network.parameters(), lr=0.0001)

      #Interior Loss
      interior_tensor = torch.from_numpy(interior_data).float()
      int_output = network(interior_tensor, requires_grad=True)
      grad_vec = np.zeros((interior_data.shape[0],2))

      interior_loss_1 = 0
      interior_loss_2 = 0

      #First Term
      interior_loss_1 += -int_output[:,0]
      interior_loss_2 += -int_output[:,1]

      #Second Term
      for j in range(int_output.shape[1]):
        for i in range(int_output.shape[0]):
          g = int_output[i,j]
          g.backward(retain_graph=True)
          gradient_input = interior_tensor.grad
          grad_vec[i,j] = gradient_input[i,1]
          optimizer.zero_grad()
          x.grad.data = torch.full((interior_data.shape[0],3),0, requires_grad = True, dtype = torch.float32)

      interior_loss_1 += interior_tensor[:,2].detach().numpy() * grad_vec[:,0]
      interior_loss_2 += interior_tensor[:,2].detach().numpy() * grad_vec[:,1]

      #Third Term
      scatter_integral_1 = 0
      scatter_integral_2 = 0

      interior_loss_1 += scatter_integral_1
      interior_loss_2 += scatter_integral_2


      #interior_loss =

      #Boundary Loss
      criterion = nn.MSELoss()

      boundary_tensor_1 = torch.from_numpy(boundary_data_1).float()
      boundary_output_1 = network(boundary_tensor_1, requires_grad=True)
      target_1 = torch.full((boundary_output_1[0],2),0)
      boundary_loss_1 = criterion(boundary_output_1, target)
      optimizer.zero_grad()
      boundary_tensor_2 = torch.from_numpy(boundary_data_2).float()
      boundary_output_2 = network(boundary_tensor_2, requires_grad=True)
      target_2 = torch.full((boundary_output_2[0],2),1)
      boundary_loss_2 = criterion(boundary_output_2, target)
