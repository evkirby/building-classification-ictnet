from model import build_model
from data_loader import get_data_loader

data_path = '..\..\AerialImageDataset'

train_generator, valid_generator, steps_per_epoch, validation_steps = get_data_loader(data_path)

model = build_model()

# A utiliser pour le fit(x, y, callbacks=[keras.callbacks.ModelCheckpoint(filepath, ...)
model.fit(train_generator, epochs=10,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          validation_data=valid_generator)