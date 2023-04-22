#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install transformers datasets evaluate


# In[1]:


import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW


# In[2]:


from huggingface_hub import notebook_login

notebook_login()


# In[ ]:


from datasets import load_dataset

food = load_dataset("food101", split="train[:5000]")


# In[1]:


food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}


# In[2]:


labels = food["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


# In[3]:


id2label[str(79)]
'prime_rib'


# In[5]:


from tensorflow import keras
from tensorflow.keras import layers

size = (image_processor.size["height"], image_processor.size["width"])

train_data_augmentation = keras.Sequential(
    [
        layers.RandomCrop(size[0], size[1]),
        layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="train_data_augmentation",
)

val_data_augmentation = keras.Sequential(
    [
        layers.CenterCrop(size[0], size[1]),
        layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ],
    name="val_data_augmentation",
)


# In[6]:


import numpy as np
import tensorflow as tf
from PIL import Image


def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # `expand_dims()` is used to add a batch dimension since
    # the TF augmentation layers operates on batched inputs.
    return tf.expand_dims(tf_image, 0)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch


# In[7]:


food["train"].set_transform(preprocess_train)
food["test"].set_transform(preprocess_val)


# In[8]:


from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train the model
for epoch in range(2):
    for batch in train_loader:
        input_ids = pad_sequence([torch.tensor(x) for x in batch["input_ids"]], batch_first=True).to(device)
        attention_mask = pad_sequence([torch.tensor(x) for x in batch["attention_mask"]], batch_first=True).to(device)
        labels = torch.tensor(batch["label"]).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


# In[9]:


import evaluate

accuracy = evaluate.load("accuracy")


# In[10]:


#create a function that passes your predictions and labels to compute to calculate the accuracy:
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# In[11]:


from transformers import create_optimizer

batch_size = 16
num_epochs = 5
num_train_steps = len(food["train"]) * num_epochs
learning_rate = 3e-5
weight_decay_rate = 0.01

optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=0,
)


# In[12]:



from transformers import TFAutoModelForImageClassification

model = TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)


# In[13]:


# converting our train dataset to tf.data.Dataset
tf_train_dataset = food["train"].to_tf_dataset(
    columns=["pixel_values"], label_cols=["label"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

# converting our test dataset to tf.data.Dataset
tf_eval_dataset = food["test"].to_tf_dataset(
    columns=["pixel_values"], label_cols=["label"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)


# In[14]:


from tensorflow.keras.losses import SparseCategoricalCrossentropy

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)


# In[15]:


from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
push_to_hub_callback = PushToHubCallback(
    output_dir="food_classifier",
    tokenizer=image_processor,
    save_strategy="no",
)
callbacks = [metric_callback, push_to_hub_callback]


# In[17]:


model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)
#Epoch 1/5
#250/250 [==============================] - 313s 1s/step - loss: 2.5623 - val_loss: 1.4161 - accuracy: 0.9290
#Epoch 2/5
#250/250 [==============================] - 265s 1s/step - loss: 0.9181 - val_loss: 0.6808 - accuracy: 0.9690
#Epoch 3/5
#250/250 [==============================] - 252s 1s/step - loss: 0.3910 - val_loss: 0.4303 - accuracy: 0.9820
#Epoch 4/5
#250/250 [==============================] - 251s 1s/step - loss: 0.2028 - val_loss: 0.3191 - accuracy: 0.9900
#Epoch 5/5
#250/250 [==============================] - 238s 949ms/step - loss: 0.1232 - val_loss: 0.3259 - accuracy: 0.9890


# In[18]:


ds = load_dataset("food101", split="validation[:10]")
image = ds["image"][0]


# In[19]:


from transformers import pipeline

classifier = pipeline("image-classification", model="my_awesome_food_model")
classifier(image)
[{'score': 0.31856709718704224, 'label': 'beignets'},
 {'score': 0.015232225880026817, 'label': 'bruschetta'},
 {'score': 0.01519392803311348, 'label': 'chicken_wings'},
 {'score': 0.013022331520915031, 'label': 'pork_chop'},
 {'score': 0.012728818692266941, 'label': 'prime_rib'}]


# In[20]:


from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("MariaK/food_classifier")
inputs = image_processor(image, return_tensors="tf")


# In[39]:


from transformers import TFAutoModelForImageClassification

model = TFAutoModelForImageClassification.from_pretrained("MariaK/food_classifier")
logits = model(**inputs).logits


# In[21]:


predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
model.config.id2label[predicted_class_id]
'beignets'


# In[ ]:




