import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # don't show figure, only save it

def plotGraph(prediction, gt):
    line_x = 3338
    fig1 = plt.figure(figsize=(12, 8))
    plt.plot(prediction, color='red', label='Prediction', linewidth=0.5)
    plt.plot(gt, color='blue', label='Answer', linewidth=0.5)
    plt.axvline(x=line_x) # before line: training phase, after line: testing phase
    plt.legend(loc='best')
    plt.title("ETF50 Stock Price Prediction (all)")
    plt.savefig("./graph/graph_all.png")

    fig1 = plt.figure(figsize=(12, 8))
    plt.plot(prediction[line_x:], color='red', label='Prediction', linewidth=0.5)
    plt.plot(gt[line_x:], color='blue', label='Answer', linewidth=0.5)
    plt.legend(loc='best')
    plt.title("ETF50 Stock Price Prediction (testing)")
    plt.savefig("./graph/graph_testing.png")


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_path'],
        config['data_loader']['args']['howManyDays'],
        testing_split = config['data_loader']['args']['testing_split'],
        normalize_info_path = config['data_loader']['args']['normalize_info_path'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    with open( config['data_loader']['args']['normalize_info_path'], 'rb') as f:
        stat = pickle.load( f)


    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    prediction = []
    gt = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            prediction += (torch.squeeze(output) * stat["std"].close + stat["mean"].close).tolist()
            gt += (torch.squeeze(target) * stat["std"].close + stat["mean"].close).tolist()

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


    plotGraph(prediction, gt)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
