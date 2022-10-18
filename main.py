# -*- coding: utf-8 -*-
"
import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import sys
from datetime import datetime

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

import model
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--trainds", type=str, default="", help="input dataset path")
    parser.add_argument("--valds", type=str, default="", help="input dataset path")
    parser.add_argument("--testds", type=str, default="", help="input dataset path")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--product", type=str, default="")
    parser.add_argument("--epochs", type=str, default="")
    parser.add_argument("--learning_rate", type=str, default="")
    parser.add_argument("--batch_size", type=str, default="")
    parser.add_argument("--data_augmentation_flag", type=str, default="")
    parser.add_argument("--earlystopper_patience", type=str, default="")
    parser.add_argument("--lr_reducer_patience", type=str, default="")
    parser.add_argument("--outputpath", type=str, default="", help="input dataset path")
    parser.add_argument("--mlfuri", type=str, default="")
    parser.add_argument("--mlfexpid", type=str, default="")
    parser.add_argument("--mlfrunid", type=str, default="")

    opt = parser.parse_args()

    mlflow.set_tracking_uri(opt.mlfuri)

    mlflow.start_run(
        run_id=opt.mlfrunid,
        experiment_id=opt.mlfexpid,
        tags={"version": opt.tag},
    )

    mlflow.log_param("epochs", opt.epochs)
    mlflow.log_param("learing_rate", opt.learning_rate)
    mlflow.log_param("batch_size", opt.batch_size)

    product = opt.product

    train_images = utils.images_data_loader(opt.trainds + product + "_train_images.h5")
    train_masks = utils.masks_data_loader(opt.trainds + product + "_train_masks.h5")
    
    val_images = utils.images_data_loader(opt.valds + product + "_val_images.h5")
    val_masks = utils.masks_data_loader(opt.valds + product + "_val_masks.h5")

    test_images = utils.images_data_loader(opt.testds + product + "_test_images.h5")
    test_masks = utils.masks_data_loader(opt.testds + product + "_test_masks.h5")

    train_masks = np.expand_dims(train_masks, -1)
    val_masks = np.expand_dims(val_masks, -1)
    test_masks = np.expand_dims(test_masks, -1)

    print("TRAIN SET")
    print(train_images.shape)
    print(train_masks.shape)
    print("VAL SET")
    print(val_images.shape)
    print(val_masks.shape)
    print("TEST SET")
    print(test_images.shape)
    print(test_masks.shape)

    if opt.data_augmentation_flag:

        transformed_images, transformed_masks = utils.data_augmentation(train_images, train_masks)
        # cancatenate data
        print("TRAIN SET after augmentation")
        train_images = np.concatenate((train_images, transformed_images), axis=0)
        train_masks = np.concatenate((train_masks, transformed_masks), axis=0)
        print(f"Images: {train_images.shape}")
        print(f"Masks: {train_masks.shape}")

    model_path = opt.outputpath + "/best_model.h5"

    model = model.model_generation(learning_rate=float(opt.learning_rate))

    checkpointer = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    
    earlystopper = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-5,
        patience=int(opt.earlystopper_patience),
        verbose=1,
        restore_best_weights=True,
    )

    lr_reducer = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=int(opt.lr_reducer_patience),
        verbose=1,
        min_delta=1e-5
    )

    mlflow.tensorflow.autolog()
    train_results = model.fit(
        train_images,
        train_masks / 255,
        validation_data=(val_images, val_masks / 255),
        epochs=int(opt.epochs),
        batch_size=int(opt.batch_size),
        callbacks=[checkpointer, earlystopper, lr_reducer],
    )

    model.save(
        opt.outputpath + "/building_segmentation_model.h5",
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )

    print("Evaluate on test data")
    test_results = model.evaluate(test_images, test_masks / 255, verbose=1)
    print(
        "test loss, test iou_coef, test_precision, test_recall, test_f1_score:",
        test_results,
    )
    mlflow.log_metric("test_loss", test_results[0])
    mlflow.log_metric("test_iou_coef", test_results[1])
    mlflow.log_metric("test_precision", test_results[2])
    mlflow.log_metric("test_recall", test_results[3])
    mlflow.log_metric("test_f1_score", test_results[4])
