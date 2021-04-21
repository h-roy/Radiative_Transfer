
#Custom Loss to minimize the residual on the interior and boudary data

class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pinn, interior_data, boundary_data_1, boundary_data_2, lambda_loss):

      criterion = nn.MSELoss()
      #Interior Loss
      x = torch.from_numpy(interior_data).float()
      xc = x.clone()
      xc.requires_grad = True
      upred = pinn(xc)
      upred = upred.to(torch.float32)

      pde_lhs_1 = 0
      pde_lhs_2 = 0

      #First Term
      pde_lhs_1 += -upred[:,0]
      pde_lhs_2 += -upred[:,1]

      #Second Term

      u_pred_grad_1 = torch.autograd.grad(upred[:,0].sum(),xc,create_graph=True, allow_unused=True)[0]
      u_pred_grad_2 = torch.autograd.grad(upred[:,1].sum(),xc,create_graph=True, allow_unused=True)[0]
      pde_lhs_1 += x[:,2] * u_pred_grad_1[:,1]
      pde_lhs_2 += x[:,2] * u_pred_grad_2[:,1]

      #Third Term

      #Interior Loss:
      ic_target_1 = torch.zeros(pde_lhs_1.shape[0], dtype=torch.float32)
      ic_target_2 = torch.zeros(pde_lhs_2.shape[0], dtype=torch.float32)
      interior_loss_1 = criterion(pde_lhs_1, ic_target_1)
      interior_loss_2 = criterion(pde_lhs_2, ic_target_2)

      #Boundary Loss


      x_bc_1 = torch.from_numpy(boundary_data_1).float()
      upred_bc_1 = pinn(x_bc_1)
      upred_bc_1 = upred_bc_1.to(torch.float32)
      bc_target_1 = torch.full((upred_bc_1.shape[0],2),0, dtype=torch.float32)
      boundary_loss_1 = criterion(upred_bc_1, bc_target_1)

      x_bc_2 = torch.from_numpy(boundary_data_2).float()
      upred_bc_2 = pinn(x_bc_2)
      upred_bc_2 = upred_bc_2.to(torch.float32)
      bc_target_2 = torch.full((upred_bc_2.shape[0],2),1, dtype=torch.float32)
      boundary_loss_2 = criterion(upred_bc_2, bc_target_2)

      loss = interior_loss_1 + interior_loss_2 + lambda_loss * (boundary_loss_1 + boundary_loss_2)

      return loss
