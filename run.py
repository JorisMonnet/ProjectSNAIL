import hydra
import wandb
from hydra.utils import instantiate
from math import ceil
from omegaconf import OmegaConf
from prettytable import PrettyTable

from datasets.cell.tabula_muris import *
from utils.io_utils import get_resume_file, hydra_setup, fix_seed, model_to_dict, opt_to_dict, get_model_file


def initialize_dataset_model(cfg):
    # Instantiate train dataset as specified in dataset config under simple_cls or set_cls
    if cfg.method.type == "baseline":
        train_dataset = instantiate(cfg.dataset.simple_cls, batch_size=cfg.method.train_batch, mode='train')
    elif cfg.method.type == "meta":
        train_dataset = instantiate(cfg.dataset.set_cls, mode='train')
    else:
        raise ValueError(f"Unknown method type: {cfg.method.type}")
    train_loader = train_dataset.get_data_loader()

    # Instantiate val dataset as specified in dataset config under simple_cls or set_cls
    # Eval type (simple or set) is specified in method config, rather than dataset config
    if cfg.method.eval_type == 'simple':
        val_dataset = instantiate(cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode='val')
    else:
        val_dataset = instantiate(cfg.dataset.set_cls, mode='val')
    val_loader = val_dataset.get_data_loader()

    # For MAML (and other optimization-based methods), need to instantiate backbone layers with fast weight
    if cfg.method.fast_weight:
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim, fast_weight=True)
    else:
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim)

    # Instantiate few-shot method class
    model = instantiate(cfg.method.cls, backbone=backbone)

    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.method.name == 'maml':
        cfg.method.stop_epoch *= model.n_task  # maml use multiple tasks in one update

    return train_loader, val_loader, model


@hydra.main(version_base=None, config_path='conf', config_name='main')
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if "name" not in cfg.exp:
        raise ValueError("The 'exp.name' argument is required!")

    if cfg.mode not in ["train", "test"]:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    fix_seed(cfg.exp.seed)

    train_loader, val_loader, model = initialize_dataset_model(cfg)

    if cfg.mode == "train":
        model = train(train_loader, val_loader, model, cfg)

    results = []
    print("Checkpoint directory:", cfg.checkpoint.dir)
    for split in cfg.eval_split:
        acc_mean, acc_std = test(cfg, model, split)
        results.append([split, acc_mean, acc_std])

    print(f"Results logged to ./checkpoints/{cfg.exp.name}/results.txt")

    if cfg.mode == "train":
        table = wandb.Table(data=results, columns=["split", "acc_mean", "acc_std"])
        wandb.log({"eval_results": table})

    display_table = PrettyTable(["split", "acc_mean", "acc_std"])
    for row in results:
        display_table.add_row(row)

    print(display_table)


def train(train_loader, val_loader, model, cfg):
    cfg.checkpoint.time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # add short date and time to checkpoint dir
    # cfg.checkpoint.dir += f"/{cfg.checkpoint.time}"

    cp_dir = os.path.join(cfg.checkpoint.dir, cfg.checkpoint.time)

    if not os.path.isdir(cp_dir):
        os.makedirs(cp_dir)
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True),
               group=cfg.exp.name, settings=wandb.Settings(start_method="thread"), mode=cfg.wandb.mode)
    wandb.define_metric("*", step_metric="epoch")

    if cfg.exp.resume:
        resume_file = get_resume_file(cp_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            cfg.method.start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])

    optimizer = instantiate(cfg.optimizer_cls, params=model.parameters())

    print("Model Architecture:")
    print(model)
    wandb.config.update({"model_details": model_to_dict(model)})

    print("Optimizer:")
    print(optimizer)
    wandb.config.update({"optimizer_details": opt_to_dict(optimizer)})

    max_acc = -1

    for epoch in range(cfg.method.start_epoch, cfg.method.stop_epoch):
        wandb.log({'epoch': epoch})
        model.train()
        model.train_loop(epoch, train_loader, optimizer)

        if epoch % cfg.exp.val_freq == 0 or epoch == cfg.method.stop_epoch - 1:
            model.eval()
            acc = model.test_loop(val_loader)
            print(f"Epoch {epoch}: {acc:.2f}")
            wandb.log({'acc/val': acc})

            if acc > max_acc:
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(cp_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch % cfg.exp.save_freq == 0 or epoch == cfg.method.stop_epoch - 1:
            outfile = os.path.join(cp_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


def test(cfg, model, split):
    if cfg.method.eval_type == 'simple':
        test_dataset = instantiate(cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode=split)
    else:
        test_dataset = instantiate(cfg.dataset.set_cls, n_episode=cfg.iter_num, mode=split)

    test_loader = test_dataset.get_data_loader()

    model_file = get_model_file(cfg)

    model.load_state_dict(torch.load(model_file)['state'])
    model.eval()

    if cfg.method.eval_type == 'simple':
        acc_all = []

        num_iters = ceil(cfg.iter_num / len(test_dataset.get_data_loader()))
        cfg.iter_num = num_iters * len(test_dataset.get_data_loader())
        print("num_iters", num_iters)
        for i in range(num_iters):
            acc_mean, acc_std = model.test_loop(test_loader, return_std=True)
            acc_all.append(acc_mean)

        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

    else:
        # Don't need to iterate, as this is accounted for in num_episodes of set data-loader
        acc_mean, acc_std = model.test_loop(test_loader, return_std=True)

    with open(f'./checkpoints/{cfg.exp.name}/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = '%s-%s-%s-%s %sshot %sway' % (
            cfg.dataset.name, split, cfg.model, cfg.method.name, cfg.n_shot, cfg.n_way)

        acc_str = '%4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(cfg.iter_num))
        f.write('Time: %s, Setting: %s, Acc: %s, Model: %s \n' % (timestamp, exp_setting, acc_str, model_file))

    return acc_mean, acc_std


if __name__ == '__main__':
    hydra_setup()
    run()
    wandb.finish()
