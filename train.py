import wandb

import lit_vit as lit_vit

def train_main(init=True):
    
    args = lit_vit.utilities.misc.ret_args()

    dm, trainer, model = lit_vit.utilities.misc.environment_loader(args)
    print(args, str(model.backbone.configuration))

    trainer.fit(model, dm)

    dm.setup('test')
    trainer.test(test_dataloaders=dm.test_dataloader())

    if init:
        wandb.finish() 

if __name__ == '__main__':
    train_main()
