import argparse
import utils

parser = argparse.ArgumentParser(
    description="Prints out training loss, validation loss and validation accuracy as the network trains")

parser.add_argument('data_dir', nargs='*', action='store', default="./flowers")
parser.add_argument('--save_dir', dest='save_dir', action='store', default='./checkpoint.pth')
parser.add_argument('--dropout_prob', dest='drop_prob', action='store', default = 0.2)
parser.add_argument('--arch', dest="arch", action='store', default='vgg16')
parser.add_argument('--learning_rate', dest='lr', action='store', default=0.01)
parser.add_argument('--hidden_units', dest='hidden_units', action='store', default=1024)
parser.add_argument('--epochs', dest='epochs', action='store', default=1)
parser.add_argument('--gpu', dest='gpu', action='store', default='gpu')

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
dropout_prob = args.drop_prob
arch = args.arch
learning_rate = args.lr
hidden_units = args.hidden_units
gpu = args.gpu
epochs = args.epochs

trainloader, validloader, testloader = utils.data_load(data_dir)

model, criterion, optimizer = utils.create_network(arch, learning_rate, hidden_units, dropout_prob)

model = utils.train_network(model, criterion, optimizer, trainloader, validloader, epochs, gpu)

utils.save_checkpoint(model, data_dir, save_dir, arch, hidden_units, dropout_prob, learning_rate, epochs)


