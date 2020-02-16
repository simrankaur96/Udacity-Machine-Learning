import argparse
import utils
import json
import numpy as np

parser = argparse.ArgumentParser(description = 
                                 "Predictions the probability, and name of flower image given")

parser.add_argument('pathToImage', nargs='*', action='store', default='./flowers/test/101/image_07949.jpg')
parser.add_argument('checkpoint', nargs='*', action='store', default='./checkpoint.pth')
parser.add_argument('--gpu', dest='gpu', action='store', default='gpu')
parser.add_argument('--top_k', dest='topk', action='store', default=1, type=int)
parser.add_argument('--category_names', dest='category', action='store', default='./cat_to_name.json')

args = parser.parse_args()
pathToImage = args.pathToImage
checkpoint = args.checkpoint
gpu = args.gpu
topk = args.topk
cat_name = args.category

model = utils.load_checkpoint(checkpoint)
probabilities = utils.predict(pathToImage, model, topk, gpu)

with open(cat_name, 'r') as json_file:
    cat_to_name = json.load(json_file)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < topk:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1
