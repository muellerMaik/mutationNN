from .enum_Class import Activation_Function
from math import exp, tanh, sinh

class Neuron:
    def __init__(self, ID: int, LAYER: int, CATEGORY: str,  ACTIVATION: Activation_Function, BIAS: float) -> None:
         self.ID = ID
         self.LAYER = LAYER
         self.CATEGORY = CATEGORY
         self.ACTIVATION = ACTIVATION
         self.BIAS = BIAS
    
         self.VALUE = 0.0

    def update_Value(self, VALUE: float) -> None:
        self.VALUE = VALUE

    def add_Value(self, VALUE: float, WEIGHT: float) -> None:
        self.VALUE += VALUE * WEIGHT
        
    def add_Bias(self):
        self.VALUE += self.BIAS

    def apply_Activation(self) -> None:
        match self.ACTIVATION:
            case Activation_Function.LINEAR:
                pass
            case Activation_Function.RELU:
                if self.VALUE < 0:
                    self.VALUE = 0
            case Activation_Function.SQUARE:
                self.VALUE *= self.VALUE
            case Activation_Function.ROOT:
                # By using the Power of 0.5, we avoid a ValueError that would be caused by sqrt() in the case of negative values.
                self.VALUE = abs(self.VALUE)**0.5
            case Activation_Function.SIGMOID:
                self.VALUE = 1/(1 + exp(-self.VALUE))
            case Activation_Function.TANH:
                self.VALUE = tanh(self.VALUE)
            case Activation_Function.SINH:
                self.VALUE = sinh(self.VALUE)
                
    def mutate_Bias(self, mutation_Value: float):
        self.BIAS *= mutation_Value
        
    def mutate_Activation(self, mutation_Value):
        self.ACTIVATION = Activation_Function(mutation_Value)