from activations.torch import LReLU, ReLU, ActivationModule, Rational

for function in [LReLU, ReLU, Rational]:
    print(f"Instanciating {function}")
    function()

print("Will show only Lrelu instanciated functions:")
LReLU.show_all()
print("Will show all instanciated activation functions (from ActivationModule):")
ActivationModule.show_all()
# import ipdb; ipdb.set_trace()
# LReLU.show_all()
