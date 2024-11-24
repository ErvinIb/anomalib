# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Fastflow
from anomalib.engine import Engine

def main():
    # Initialize the datamodule, model and engine
    datamodule = MVTec(category='toothbrush', 
                       train_batch_size=8, 
                       eval_batch_size=8, 
                       num_workers=4, 
                       seed=0)
    model = Fastflow()
    engine = Engine(pixel_metrics=['AUROC'], image_metrics=['AUROC'])

    print('Train Batch Size: ', datamodule.train_batch_size)
    print('Eval Batch Size: ', datamodule.eval_batch_size)
    print('Num Workers: ', datamodule.num_workers)
    #print('Max Epochs: ', engine.trainer.max_epochs)
    
    # Train the model
    engine.fit(datamodule=datamodule, model=model)

    # load best model from checkpoint before evaluating
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )

    print(test_results)

    # Assuming the datamodule, model and engine is initialized from the previous step,
    # a prediction via a checkpoint file can be performed as follows:
    predictions = engine.predict(
        datamodule=datamodule,
        model=model,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )
    print(predictions)

if __name__ == '__main__':
    main()