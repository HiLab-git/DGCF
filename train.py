import time
from tqdm import tqdm

from code_dataset import create_dataset
from code_model import create_model
from code_config.parser import parse
from code_record.visualizer import Visualizer
from code_util import util
from code_network.tools.scheduler import get_num_epochs

def train(status_config = None, common_config = None):

    # opt >>>> config
    config,common_config = parse("train",status_config = status_config, common_config=common_config) 
    val_config, _ = parse("train",status_config = status_config,save=False,val=True)
    
    # random seed
    util.set_random_seed(config["random_seed"])

    # dataset
    train_loader, _ = create_dataset(config)  # create a dataset given dataset_mode and other configurations
    val_loader, val_len = create_dataset(val_config)  # create a dataset given dataset_mode and other configurations

    # model
    model = create_model(config)      # init model
    model.setup(config)               # load network for test; set scheduler for train
    
    # visualizer 
    visualizer = Visualizer(config)   
 
    total_iters = 0                # the total number of training iterations
    num_epochs = get_num_epochs(config)
    start_time = time.time()  # timer for entire training process

    use_html = config["record"].get("html",{}).get("use_html",False)
    use_tensorboard = config["record"].get("tensorboard",{}).get("use_tensorboard",False)

    # save model
    best_metric = 0

    if val_len > 0 and config.get("continue", {}).get("continue_train", False) == True: # 如果是继续训练 则在训练开始之前进行一次validation
    # if val_len > 0:
        epoch = 0
        val_start_time = time.time()  # timer for entire epoch
        val_losses = {}
        val_metrics = {}
        model.eval()
        for data in tqdm(val_loader, desc="epoch %d/%d - val" % (epoch, num_epochs), position=1, leave=False):
            model.set_input(data) 
            model.calculate_loss()          
            losses = model.get_current_losses()
            model.calclulate_metric()
            metrics = model.get_current_metrics()
            val_losses = util.merge_dicts_add_values(val_losses, losses)
            val_metrics = util.merge_dicts_add_values(val_metrics, metrics)
        val_losses_avg = util.dict_divided_by_number(val_losses,len(val_loader))
        val_metrics_avg = util.dict_divided_by_number(val_metrics,len(val_loader))
        log_info_val = f"Epoch {epoch}/{num_epochs} - Time: {time.time() - val_start_time:.2f}s - val Losses: {util.dict2str(val_losses_avg)} - val Metrics: {util.dict2str(val_metrics_avg)}"
        tqdm.write(log_info_val)
        visualizer.record_log(log_info_val, phase="val")
        visuals = model.get_current_visuals()
        if use_html:  # html
            visualizer.display_on_html(visuals, data["A"]["params"]["path"], phase = "val", epoch = epoch)
        if use_tensorboard:
            visualizer.display_on_tensorboard(model.get_current_visuals(), step = epoch, phase="val")
            visualizer.plot_scalars_on_tensorboard(val_losses_avg, epoch, phase="val")
            visualizer.plot_scalars_on_tensorboard(val_metrics_avg, epoch, phase="val")

        # if config["record"].get("CAM",{}).get("use_CAM",False):
        #    visualizer.draw_CAM(model,config,epoch = epoch)
        if config["record"]["save_model"].get("save_best", False):
            metric_name = config["record"]["save_model"].get("best_metric", "ssim")
            if metric_name in val_metrics_avg:
                metric = val_metrics_avg[metric_name]
                if metric > best_metric:
                    best_metric = metric
                    tqdm.write(f"New best {metric_name}: {best_metric:.3f} at epoch {epoch}")
                    if epoch % config["record"]["save_model"]["per_epoch"] == 0:  # cache our model every <save_epoch_freq> epochs
                    # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                        model.save_networks(f"{epoch}_{metric_name}_{best_metric:.3f}")
                    model.save_networks('best')

    for epoch in tqdm(range(1, num_epochs+1), desc="Epochs", position=0):   
        model.train()
        epoch_iter = 0 # the number of training iterations in current epoch, reset to 0 every epoch
        train_losses = {}  # record losses for current epoch
        train_metrics = {}
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", position=1, leave=False):
            total_iters += config["dataset"]["dataloader"]["batch_size"]
            epoch_iter += config["dataset"]["dataloader"]["batch_size"]
            if epoch_iter % config["record"]["record_loss_per_iter"] == 0:
                iter_start_time = time.time()  # timer for computation per iteration
                t_data = iter_start_time - iter_data_time

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            model.calclulate_metric()
            metrics = model.get_current_metrics()
            train_losses = util.merge_dicts_add_values(train_losses, losses)
            train_metrics = util.merge_dicts_add_values(train_metrics, metrics)

            if epoch_iter % config["record"]["record_loss_per_iter"] == 0:    # loss to txt 
                t_comp = time.time() - iter_start_time
                log_info_train_iter = f"Epoch {epoch}/{num_epochs} - Iter {epoch_iter} - t_comp: {t_comp:.4f}s - t_data: {t_data:.4f}s - Losses: {util.dict2str(losses)} - Metrics: {util.dict2str(metrics)}"
                tqdm.write(log_info_train_iter)
                visualizer.record_log(log_info_train_iter, phase="train")
            
            if use_html:
                if epoch_iter % config["record"]["html"]["display_per_iter"] == 0:  # html
                    visualizer.display_on_html(model.get_current_visuals(), data["A"]["params"]["path"], phase = "train", epoch = epoch, iter = epoch_iter)
            
            if use_tensorboard:
                if epoch_iter % config["record"]["tensorboard"]["display_per_iter"] == 0:
                    visualizer.display_on_tensorboard(model.get_current_visuals(), step=epoch_iter, phase="train")

            iter_data_time = time.time() # refresh the time for data loading 

        # Calculate average loss for the epoch
        t_comp = time.time() - epoch_start_time
        train_losses_avg = util.dict_divided_by_number(train_losses, len(train_loader))
        train_metrics_avg = util.dict_divided_by_number(train_metrics, len(train_loader))
        if use_tensorboard == True:
            visualizer.plot_scalars_on_tensorboard(train_losses_avg, epoch, phase="train")
            visualizer.plot_scalars_on_tensorboard(train_metrics_avg, epoch, phase="train")
        log_info_train_epoch = f"Epoch {epoch}/{num_epochs} - Time: {t_comp:.2f}s - Losses: {util.dict2str(train_losses_avg)} - Metrics: {util.dict2str(train_metrics_avg)}"
        tqdm.write(log_info_train_epoch)
        visualizer.record_log(log_info_train_epoch, phase="train")  
        model.update_learning_rate()  # update learning rates

        if config["record"]["save_model"].get("save_latest", False):
            model.save_networks('latest')
        
        tqdm.write("work is going on at %s" % config["work_dir"])

        if val_len > 0:
            if epoch % config["record"]["val_per_epoch"] == 0:
                val_start_time = time.time()  # timer for entire epoch
                val_losses = {}
                val_metrics = {}
                model.eval()
                for data in tqdm(val_loader, desc="epoch %d/%d - val" % (epoch, num_epochs), position=1, leave=False):
                    model.set_input(data) 
                    model.calculate_loss()          
                    losses = model.get_current_losses()
                    model.calclulate_metric()
                    metrics = model.get_current_metrics()
                    val_losses = util.merge_dicts_add_values(val_losses, losses)
                    val_metrics = util.merge_dicts_add_values(val_metrics, metrics)
                val_losses_avg = util.dict_divided_by_number(val_losses,len(val_loader))
                val_metrics_avg = util.dict_divided_by_number(val_metrics,len(val_loader))
                log_info_val = f"Epoch {epoch}/{num_epochs} - Time: {time.time() - val_start_time:.2f}s - val Losses: {util.dict2str(val_losses_avg)} - val Metrics: {util.dict2str(val_metrics_avg)}"
                tqdm.write(log_info_val)
                visualizer.record_log(log_info_val, phase="val")
                visuals = model.get_current_visuals()
                if use_html:  # html
                    visualizer.display_on_html(visuals, data["A"]["params"]["path"], phase = "val", epoch = epoch)
                if use_tensorboard:
                    visualizer.display_on_tensorboard(model.get_current_visuals(), step = epoch, phase="val")
                    visualizer.plot_scalars_on_tensorboard(val_losses_avg, epoch, phase="val")
                    visualizer.plot_scalars_on_tensorboard(val_metrics_avg, epoch, phase="val")

                # if config["record"].get("CAM",{}).get("use_CAM",False):
                #    visualizer.draw_CAM(model,config,epoch = epoch)
                if config["record"]["save_model"].get("save_best", False):
                    metric_name = config["record"]["save_model"].get("best_metric", "ssim")
                    if metric_name in val_metrics_avg:
                        metric = val_metrics_avg[metric_name]
                        if metric > best_metric:
                            best_metric = metric
                            tqdm.write(f"New best {metric_name}: {best_metric:.3f} at epoch {epoch}")
                            if epoch % config["record"]["save_model"]["per_epoch"] == 0:  # cache our model every <save_epoch_freq> epochs
                            # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                                model.save_networks(f"{epoch}_{metric_name}_{best_metric:.3f}")
                            model.save_networks('best')
    
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = f"Total time: {total_time // 3600}h {(total_time % 3600) // 60}m {total_time % 60:.2f}s"
    tqdm.write(total_time_str)
    visualizer.record_log(total_time_str, phase="train")

    return common_config

if __name__ == '__main__':
    train()

