
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




    #def forward(self, network, x_u_train, u_train, x_b_train, u_b_train, x_f_train, x_obj, u_obj,
    #            training_ic, computing_error=False):


        #

        """lambda_residual = network.lambda_residual
        lambda_reg = network.regularization_param
        order_regularizer = network.kernel_regularizer
        space_dimensions = dataclass.space_dimensions
        BC = dataclass.BC
        solid_object = dataclass.obj

        if x_b_train.shape[0] <= 1:
            space_dimensions = 0

        u_pred_var_list = list()
        u_train_var_list = list()
        for j in range(dataclass.output_dimension):

            # Space dimensions
            # Not used for the radiative project, check at the else part
            if not training_ic and Ec.extrema_values is not None:
                for i in range(space_dimensions):
                    half_len_x_b_train_i = int(x_b_train.shape[0] / (2 * space_dimensions))

                    x_b_train_i = x_b_train[i * int(x_b_train.shape[0] / space_dimensions):(i + 1) * int(
                        x_b_train.shape[0] / space_dimensions), :]
                    u_b_train_i = u_b_train[i * int(x_b_train.shape[0] / space_dimensions):(i + 1) * int(
                        x_b_train.shape[0] / space_dimensions), :]
                    boundary = 0
                    while boundary < 2:

                        x_b_train_i_half = x_b_train_i[
                                           half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]
                        u_b_train_i_half = u_b_train_i[
                                           half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]

                        if BC[i][boundary][j] == "func":
                            u_pred_var_list.append(network(x_b_train_i_half)[:, j])
                            u_train_var_list.append(u_b_train_i_half[:, j])
                        if BC[i][boundary][j] == "der":
                            x_b_train_i_half.requires_grad = True
                            f_val = network(x_b_train_i_half)[:, j]
                            inputs = torch.ones(x_b_train_i_half.shape[0], )
                            if not computing_error and torch.cuda.is_available():
                                inputs = inputs.cuda()
                            der_f_vals = \
                                torch.autograd.grad(f_val, x_b_train_i_half, grad_outputs=inputs, create_graph=True)[0][:, i]
                            u_pred_var_list.append(der_f_vals)
                            u_train_var_list.append(u_b_train_i_half[:, j])
                        elif BC[i][boundary][j] == "periodic":
                            x_half_1 = x_b_train_i_half
                            x_half_2 = x_b_train_i[
                                       half_len_x_b_train_i * (boundary + 1):half_len_x_b_train_i * (boundary + 2), :]
                            x_half_1.requires_grad = True
                            x_half_2.requires_grad = True
                            inputs = torch.ones(x_half_1.shape[0], )
                            if not computing_error and torch.cuda.is_available():
                                inputs = inputs.cuda()
                            pred_first_half = network(x_half_1)[:, j]
                            pred_second_half = network(x_half_2)[:, j]
                            der_pred_first_half = \
                                torch.autograd.grad(pred_first_half, x_half_1, grad_outputs=inputs, create_graph=True)[
                                    0]
                            der_pred_first_half_i = der_pred_first_half[:, i]
                            der_pred_second_half = \
                                torch.autograd.grad(pred_second_half, x_half_2, grad_outputs=inputs, create_graph=True)[
                                    0]
                            der_pred_second_half_i = der_pred_second_half[:, i]

                            u_pred_var_list.append(pred_second_half)
                            u_train_var_list.append(pred_first_half)

                            u_pred_var_list.append(der_pred_second_half_i)
                            u_train_var_list.append(der_pred_first_half_i)

                            boundary = boundary + 1

                        boundary = boundary + 1
            else:
                u_pred_b, u_train_b = Ec.apply_BC(x_b_train, u_b_train, network)
                u_pred_var_list.append(u_pred_b)
                u_train_var_list.append(u_train_b)

                if Ec.time_dimensions != 0:
                    u_pred_0, u_train_0 = Ec.apply_IC(x_u_train, u_train, network)
                    u_pred_var_list.append(u_pred_0)
                    u_train_var_list.append(u_train_0)

            # Time Dimension
            if x_u_train.shape[0] != 0:
                # This is used to solve the radiative inverse problem
                try:
                    if j == 0:
                        if Ec.assign_g:
                            # print("Assign G")
                            g = Ec.get_G(network, x_u_train[:, :3], Ec.n_quad)
                            u_pred_var_list.append(g)

                        else:
                            # print("Not assign G")
                            u_pred_var_list.append(network(x_u_train)[:, j])
                        u_train_var_list.append(u_train[:, j])
                    if j == 1:
                        # print(x_b_train)
                        phys_coord_b = x_b_train[:, :3]
                        if Ec.average:
                            u_pred_var_list.append(Ec.get_average_inf_q(network, phys_coord_b, 10))
                        else:
                            u_pred_var_list.append(network(x_b_train)[:, j])
                        u_train_var_list.append(Ec.K(phys_coord_b[:, 0], phys_coord_b[:, 1], phys_coord_b[:, 2]))
                        # Compute tikonov regularization

                        x_f_train.requires_grad = True
                        k = network(x_f_train)[:, j].reshape(-1, )
                        lambda_k = 0.01

                        grad_k = torch.autograd.grad(k, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(Ec.dev), create_graph=True)[0]

                        grad_k_x = grad_k[:, 0]
                        grad_k_y = grad_k[:, 1]

                        tik_reg = torch.sqrt(lambda_k * torch.mean(grad_k_x ** 2 + grad_k_y ** 2)).reshape(1, )

                        u_pred_var_list.append(tik_reg)
                        u_train_var_list.append(torch.zeros_like(tik_reg))
                except AttributeError:
                    u_pred_var_list.append(network(x_u_train)[:, j])
                    u_train_var_list.append(u_train[:, j])

            if x_obj is not None and not training_ic:
                if BC[-1][j] == "func":
                    u_pred_var_list.append(network(x_obj)[:, j])
                    u_train_var_list.append(u_obj[:, j])

                if BC[-1][j] == "der":
                    x_obj_grad = x_obj.clone()
                    x_obj_transl = x_obj_grad[(np.arange(0, x_obj_grad.shape[0]) + 1) % (x_obj_grad.shape[0]), :]
                    x_obj_mean = (x_obj_grad + x_obj_transl) / 2
                    x_obj_mean.requires_grad = True
                    f_val = network(x_obj_mean)[:, j]
                    inputs = torch.ones(x_obj_mean.shape[0], )
                    if not computing_error and torch.cuda.is_available():
                        inputs = inputs.cuda()
                    der_f_vals_x = torch.autograd.grad(f_val, x_obj_mean, grad_outputs=inputs, create_graph=True)[0][:,
                                   0]
                    der_f_vals_y = torch.autograd.grad(f_val, x_obj_mean, grad_outputs=inputs, create_graph=True)[0][:,
                                   1]
                    t = (x_obj_grad - x_obj_transl)

                    nx = t[:, 1] / torch.sqrt(t[:, 1] ** 2 + t[:, 0] ** 2)
                    ny = -t[:, 0] / torch.sqrt(t[:, 1] ** 2 + t[:, 0] ** 2)
                    der_n = der_f_vals_x * nx + der_f_vals_y * ny
                    u_pred_var_list.append(der_n)
                    u_train_var_list.append(u_obj[:, j])

        u_pred_tot_vars = torch.cat(u_pred_var_list, 0)
        u_train_tot_vars = torch.cat(u_train_var_list, 0)

        if not computing_error and torch.cuda.is_available():
            u_pred_tot_vars = u_pred_tot_vars.cuda()
            u_train_tot_vars = u_train_tot_vars.cuda()

        assert not torch.isnan(u_pred_tot_vars).any()

        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** 2))

        if not training_ic:

            res = Ec.compute_res(network, x_f_train, space_dimensions, solid_object, computing_error)
            res_train = torch.tensor(()).new_full(size=(res.shape[0],), fill_value=0.0)

            if not computing_error and torch.cuda.is_available():
                res = res.cuda()
                res_train = res_train.cuda()

            loss_res = (torch.mean(abs(res) ** 2))

            u_pred_var_list.append(res)
            u_train_var_list.append(res_train)

        loss_reg = regularization(network, order_regularizer)
        if not training_ic:
            loss_v = torch.log10(
                loss_vars + lambda_residual * loss_res + lambda_reg * loss_reg)  # + lambda_reg/loss_reg
        else:
            loss_v = torch.log10(loss_vars + lambda_reg * loss_reg)
        print("final loss:", loss_v.detach().cpu().numpy().round(4), " ", torch.log10(loss_vars).detach().cpu().numpy().round(4), " ",
              torch.log10(loss_res).detach().cpu().numpy().round(4))
        return loss_v
"""
