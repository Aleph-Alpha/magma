from rational.torch import Rational
import torch

rat_l = Rational("leaky_relu")
rat_s = Rational("sigmoid")
rat_i = Rational("identity")

device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = torch.nn.MSELoss()

optimizers = [torch.optim.Adam(rat.parameters(), lr=0.01)
              for rat in Rational.list]

capturing_epochs = [0, 1, 2, 4, 8, 16, 32, 64, 99, 149, 199]
for epoch in range(200):
    for (rat, optimizer) in zip(Rational.list, optimizers):
        inp = ((torch.rand(10000)-0.5)*5).to(device)
        exp = torch.sin(inp)
        optimizer.zero_grad()
        out = rat(inp)
        loss = criterion(out, exp)
        loss.backward()
        optimizer.step()
    if epoch in capturing_epochs:
        Rational.capture_all(f"Epoch {epoch}")

Rational.export_evolution_graphs(other_func=torch.sin)

Rational.use_kde = False
capturing_epochs = [0, 1, 2, 4, 8, 16, 32, 64, 99, 149, 199]
for epoch in range(200):
    for (rat, optimizer) in zip(Rational.list, optimizers):
        inp = torch.cat([torch.randn(1000)+i for i in range(-3, 4, 3)]).to(device)
        exp = torch.sin(inp)
        optimizer.zero_grad()
        if epoch in capturing_epochs:
            Rational.save_all_inputs(True)
        out = rat(inp)
        loss = criterion(out, exp)
        loss.backward()
        optimizer.step()
    if epoch in capturing_epochs:
        Rational.capture_all(f"Epoch {epoch}")
        Rational.save_all_inputs(False)

Rational.export_evolution_graphs(other_func=torch.sin)
