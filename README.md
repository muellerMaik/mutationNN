# mutationNN
# mutationNN is a very simple Python Package.
# Install: pip install mutationNN
# Let's create a Neural Network with 3 Inputs and 2 Outputs:

from mutationNN import NN
My_NN = NN(
  INPUT_SIZE=3,
  OUTPUT_SIZE=2,
  MUTATION_CHANCE=0.25,
  FULLY_CONNECTED=False,
)

# Please Note that "MUTATION_CHANCE" and "FULLY_CONNECTED" have default values of 0.25 and False respectively.
# To get a Single Output use .process_Input(). It will return a List of the size OUTPUT_SIZE.
# This should be used if the output will change the next Input.

p_I = [1, 2, 3]
output = My_NN.process_Input(p_I)

# If the Output does not Change the next Input you can use .process_Static_Input().
# .process_Static_Input() takes a List of Lists as Input and returns a List of Lists as output.
# Inside the NN .process_Static_Input() just iterates through the List and calls process_Input().

p_S_I = [
  [1, 2, 3],
  [2, 3, 4],
  [3, 4, 5]
]

outputs = My_NN.process_Static_Input(p_S_I)

# To Mutate the NN use .mutate() "must_mutate" has the default value "False".
My_NN.mutate(must_mutate=False)

# To Copy and Change another NN use copy_NN(other_NN)
# It will copy the NN given as an Argument.
# Afterwards it will call .mutate(must_mutate=True) to ensure it is at least slightly different.

My_NN_2 = NN(
  INPUT_SIZE=3,
  OUTPUT_SIZE=2,
  MUTATION_CHANCE=0.25,
  FULLY_CONNECTED=False,
) 

My_NN_2.copy_NN(My_NN)
