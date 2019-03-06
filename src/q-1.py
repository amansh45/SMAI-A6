import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
import os


def convolve(inputs, filter):
	filters_dim = filter.shape
	inputs_dim = inputs.shape
	
	feature_map = [[]]

	broke = False

	for i in range(len(inputs)):
		for j in range(len(inputs[i])):
			if inputs_dim[1]-j<filters_dim[1] or inputs_dim[0]-i<filters_dim[0]:
				broke = True
				break
			else:
				window = []
				m=i
				n=j
				for p in range(filters_dim[0]):
					row = []
					for q in range(filters_dim[1]):
						row.append(inputs[m][n])
						n+=1
					n=j
					window.append(row)
					m+=1
				window = np.array(window)
				
				x = np.sum(np.multiply(window, filter))
				if broke:
					feature_map.append([x])
				else:
					feature_map[i].append(x)

				broke = False
	return np.array(feature_map)


def max_pooling(inputs, stride=2, pooling_dim=(2,2)):
	inputs_dim = inputs.shape
	i = 0
	feature_map = [[]]

	broke = False
	while i<len(inputs):
		j = 0
		while j<len(inputs[i]):
			if inputs_dim[1]-j<pooling_dim[1] or inputs_dim[0]-i<pooling_dim[0]:
				broke = True
				break
			else:
				window = []
				m=i
				n=j
				for p in range(pooling_dim[0]):
					row = []
					for q in range(pooling_dim[1]):
						row.append(inputs[m][n])
						n+=1
					n=j
					window.append(row)
					m+=1
				window = np.array(window)

				p = []
				for l in range(window.shape[2]):
					p.append([])

				for r in window:
					for c in r:
						for o in range(len(c)):
							p[o].append(c[o])
							
				x = [0] * window.shape[2]

				for r in range(len(p)):
					x[r] = max(p[r])

				if broke:
					feature_map.append([x])
				else:
					try:
						feature_map[int(i/2)].append(x)
					except IndexError:
						feature_map.append([x])
				broke = False

			j+=stride
		i+=stride

	return np.array(feature_map)


def sigmoid(inputs, derivative=False):
	sigm = 1. / (1. + np.exp(-inputs))
	if derivative:
		return sigm * (1. - sigm)
	return sigm

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def apply_convolution(inputs, filters):
	feature_maps = convolve(inputs, filters[0])
	feature_maps = np.reshape(feature_maps, (feature_maps.shape[0], feature_maps.shape[1], 1))
	feature_maps = feature_maps.tolist()
	i = 1
	while i<len(filters):
		new_map = convolve(inputs, filters[i])
		for p in range(len(new_map)):
			for q in range(len(new_map[p])):
				feature_maps[p][q].append(new_map[p][q])
		i+=1

	feature_maps = np.array(feature_maps)
	return feature_maps

def apply_activation(inputs, activation_type = 'relu'):
	if activation_type == 'relu':
		for c in range(len(inputs)):
			for x in range(len(inputs[c])):
				for y in range(len(inputs[c][x])):
					inputs[c][x][y] = max(inputs[c][x][y], 0)
		return inputs

def generate_input_layer(feature_maps):
	num_maps = np.random.rand(feature_maps.shape[2],feature_maps.shape[0], feature_maps.shape[1])
	for x in range(len(feature_maps)):
		for y in range(len(feature_maps[x])):
			for z in range(len(feature_maps[x][y])):
				num_maps[z][x][y] = feature_maps[x][y][z]
	inputs = []
	for channel in num_maps:
		for rows in channel:
			for cols in rows:
				inputs.append(cols)
	return np.array(inputs)

def initialize_filters(num_filters, dimensions):
	filters = []
	for f in range(num_filters):
		filters.append(np.random.rand(dimensions[0],dimensions[1],dimensions[2]))
	return filters


def show_images(feature_maps, cols = 3, titles = None):
    num_maps = np.random.rand(feature_maps.shape[2],feature_maps.shape[0], feature_maps.shape[1])
    for x in range(len(feature_maps)):
    	for y in range(len(feature_maps[x])):
    		for z in range(len(feature_maps[x][y])):
    			num_maps[z][x][y] = feature_maps[x][y][z]

    assert((titles is None) or (len(num_maps) == len(titles)))
    n_images = len(num_maps)
    if titles is None: titles = ['convolve using filter (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(num_maps, titles)):
    	a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
    	if image.ndim == 2:
    		plt.gray()
    	plt.imshow(image)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def process_image(image):
	if len(image.shape) == 2:
		new_image = np.reshape(image, (image.shape[0],image.shape[1],1))
		return new_image
	else:
		return image

path = os.getcwd()
l = path.split('/')
img_path = l[0:len(l)-1]
img_path = '/'.join(img_path)
img_path = img_path + '/input_data/5.png'

image = img.imread(img_path)
print("Shape of the input image is: ",image.shape)
if len(image.shape) == 2:
	filters = initialize_filters(6, (5,5,1))
else:
	filters = initialize_filters(6, (5,5,image.shape[2]))
image = process_image(image)
feature_maps = apply_convolution(image, filters)
act_feature_maps = apply_activation(feature_maps, activation_type='relu')
print("After applying first convolution and activation, shape is: ",act_feature_maps.shape)
show_images(act_feature_maps)

pooled_feature_maps = max_pooling(act_feature_maps, stride = 2, pooling_dim = (2,2))
show_images(pooled_feature_maps)
print("After applying first pooling, shape is: ",pooled_feature_maps.shape)

filters = initialize_filters(16,(5,5,6))
feature_maps = apply_convolution(pooled_feature_maps, filters)
act_feature_maps = apply_activation(feature_maps, activation_type='relu')
show_images(act_feature_maps)
print("After applying second convolution and activation, shape is: ",act_feature_maps.shape)

pooled_feature_maps = max_pooling(act_feature_maps, stride = 2, pooling_dim = (2,2))
show_images(pooled_feature_maps)
print("After applying second pooling, shape is: ",pooled_feature_maps.shape)

filters = initialize_filters(120,(4,4,16))
feature_maps = apply_convolution(pooled_feature_maps, filters)
act_feature_maps = apply_activation(feature_maps, activation_type='relu')
print("After applying third convolution and activation, shape is: ",act_feature_maps.shape)

input_to_ann = generate_input_layer(act_feature_maps)

print("Input length is: ",len(input_to_ann))

weight_i_h0 = np.random.rand(len(input_to_ann), 84)
weight_h0_o = np.random.rand(84,10)

activation_h0 = sigmoid(np.dot(input_to_ann.T, weight_i_h0))
activation_o = softmax(np.dot(activation_h0.T, weight_h0_o))
print(activation_o)