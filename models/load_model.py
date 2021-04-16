
def load(model_name, input_img_size,
         num_classes,
         random_weights=False,
         unfreeze_layers=False,
         augmentation_layer=None):

    if model_name == 'mobilenet':
        from models.mobilenet import load_model
        model = load_model(input_image_size=input_img_size,
                           num_classes=num_classes,
                           random_weights=random_weights,
                           unfreeze_layers=unfreeze_layers,
                           augmentation_layer=augmentation_layer)
        return model
    else:
        raise NotImplementedError


