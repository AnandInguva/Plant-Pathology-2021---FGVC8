
def load(model_name, input_img_size,
         num_classes,
         random_weights=False,
         unfreeze_layers=False,):

    if model_name == 'mobilenet':
        from models.mobilenet import load_model
        model = load_model(input_image_size=input_img_size,
                           num_classes=num_classes,
                           random_weights=random_weights,
                           unfreeze_layers=unfreeze_layers)
        return model
    else:
        raise NotImplementedError


