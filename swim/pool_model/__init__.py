"""The pool segmentation network: finds the pool's water in a frame (docs/pool-model.md).

    network.py   the U-Net (ImageNet-pretrained ResNet encoder)
    data.py      training frames and masks from the labeling dataset, augmentation, the split
    train.py     training and validation (python -m swim.pool_model train)
    predict.py   a trained model: class per pixel of a frame

Everything here imports torch: import it inside the function that creates the model (as for the other models), so
the pipeline loads fast without it.
"""
