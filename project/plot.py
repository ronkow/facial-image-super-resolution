def file_to_list(filepath):
    list_final= []
    with open(filepath, encoding = 'utf-8') as f:
        l = f.read().splitlines()
        for i in l:
            i = i.strip()
            list_final.append(i)
    print(len(list_final))
    return list_final


def extract_loss(list_raw_data):
    loss_10k = []
    iter_loss_10k = []
    loss_rest = []
    iter_loss_rest = []
    
    for k,i in enumerate(list_raw_data):
        #print(k)
        i = json.loads(i)  # convert string to dict
        
        if 'loss' in i:
            if i['iter']%100 == 0:  # extract loss for every 100 iterations
                if i['iter'] <= 10000:
                    iter_loss_10k.append(i['iter'])
                    loss_10k.append(i['loss'])
                else:
                    iter_loss_rest.append(i['iter'])
                    loss_rest.append(i['loss'])
                    
    print('Length of loss list (<= 10k): ',len(loss_10k))
    print('Length of iter loss list (<= 10k): ',len(iter_loss_10k))
    print('Length of loss list (> 10k): ',len(loss_rest))
    print('Length of iter loss list (> 10k): ',len(iter_loss_rest))
    
    return loss_10k, iter_loss_10k, loss_rest, iter_loss_rest


def extract_psnr(list_raw_data):    
    psnr_10k = []
    iter_psnr_10k = []
    psnr_rest = []
    iter_psnr_rest = []
    
    for k,i in enumerate(list_raw_data):
        #print(k)
        i = json.loads(i)
        if 'PSNR' in i:
            if i['iter'] <= 10000:
                iter_psnr_10k.append(i['iter'])
                psnr_10k.append(i['PSNR'])
            else:
                iter_psnr_rest.append(i['iter'])
                psnr_rest.append(i['PSNR'])

    print('Length of psnr list (<= 10k): ',len(psnr_10k))
    print('Length of iter psnr list (<= 10k): ',len(iter_psnr_10k))
    print('Length of psnr list (> 10k): ',len(psnr_rest))
    print('Length of iter psnr list (> 10k): ',len(iter_psnr_rest))
    
    return psnr_10k, iter_psnr_10k, psnr_rest, iter_psnr_rest


def plot_curve(xdata, ydata, xlab, ylab, xsize, ysize, title, filename):
    plt.figure(figsize=(xsize,ysize))
    plt.xlabel(xlab, fontsize=24)
    plt.ylabel(ylab, fontsize=24)
    plt.title(title, fontsize=24)
    plt.plot(xdata, ydata)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':

    json_logs_srresnet = './srresnet_logs/srresnet.json'
    list_resnet = file_to_list(json_logs_srresnet)

    resnet_loss_10k, resnet_iter_loss_10k, resnet_loss_rest, resnet_iter_loss_rest = extract_loss(list_resnet)
    resnet_psnr_10k, resnet_iter_psnr_10k, resnet_psnr_rest, resnet_iter_psnr_rest = extract_psnr(list_resnet)


    json_logs_rrdbnet = './rrdbnet_logs/rrdbnet.json'
    list_rrdb = file_to_list(json_logs_rrdbnet)

    rrdb_loss_10k, rrdb_iter_loss_10k, rrdb_loss_rest, rrdb_iter_loss_rest = extract_loss(list_rrdb)
    rrdb_psnr_10k, rrdb_iter_psnr_10k, rrdb_psnr_rest, rrdb_iter_psnr_rest = extract_psnr(list_rrdb)


    plot_curve(resnet_iter_loss_10k, resnet_loss_10k, 'Iteration', 'Loss', 9, 6,'SRResNet Loss Curve','resnet_loss_10k.png')
    plot_curve(resnet_iter_loss_rest, resnet_loss_rest, 'Iteration', 'Loss', 16, 6,'SRResNet Loss Curve','resnet_loss_rest.png')

    plot_curve(resnet_iter_psnr_10k, resnet_psnr_10k, 'Iteration', 'PSNR', 9, 6,'SRResNet PSNR Curve','resnet_psnr_10k.png')
    plot_curve(resnet_iter_psnr_rest, resnet_psnr_rest, 'Iteration', 'PSNR', 16, 6,'SRResNet PSNR Curve','resnet_psnr_rest.png')

    plot_curve(rrdb_iter_loss_10k, rrdb_loss_10k, 'Iteration', 'Loss', 8, 6,'RRDBNet Loss Curve','rrdb_loss_10k.png')
    plot_curve(rrdb_iter_loss_rest, rrdb_loss_rest, 'Iteration', 'Loss', 16, 6,'RRDBNet Loss Curve','rrdb_loss_rest.png')

    plot_curve(rrdb_iter_psnr_10k, rrdb_psnr_10k, 'Iteration', 'PSNR', 8, 6,'RRDBNet PSNR Curve','rrdb_psnr_10k.png')
    plot_curve(rrdb_iter_psnr_rest, rrdb_psnr_rest, 'Iteration', 'PSNR', 16, 6,'RRDBNet PSNR Curve','rrdb_psnr_rest.png')