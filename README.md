# CapsNet
My implementation of CapsNet using Tensorflow

# How to use
To create a capsule object:
capsule = Capsule(num_capsules, capsule_dimension)

To get capsule output:
capsule_output, capsule_probabilities = capsule(input_tensor)

input_tensor should have shape [batch_size, n_capsules_in, dim_capsule_in]
capsule_output shape will be [batch_size, num_capsules, capsule_dimension]
capsule_probabilities have the shape [batch_size, num_capsules]
capsule_probabilities are the norms of the capsule_output vectors
