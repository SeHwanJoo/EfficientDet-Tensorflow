import argparse
from datetime import date
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam

from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientdet
from losses import smooth_l1, focal, smooth_l1_quad
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def create_callbacks(prediction_model, validation_generator, args, step1=False):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    if step1:
        earlyStopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False
        )
        callbacks.append(earlyStopping)
    #
    # def scheduler(epoch):
    #     step = min(14041, model.optimizer.lr.decay_steps)
    #     print('---------------------------------------------------------')
    #     print(((model.optimizer.lr.initial_learning_rate - model.optimizer.lr.end_learning_rate) *
    #            int((1 - step / model.optimizer.lr.decay_steps) // 1) ^ model.optimizer.lr.power
    #            ) + model.optimizer.lr.end_learning_rate)
    #     return ((model.optimizer.lr.initial_learning_rate - model.optimizer.lr.end_learning_rate) *
    #             ((1 - step / model.optimizer.lr.decay_steps) // 1) ^ model.optimizer.lr.power
    #             ) + model.optimizer.lr.end_learning_rate
    #     # tf.summary.scalar('learning rate', data=lr, step=epoch)
    #
    # lr_callbacks = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)
    # callbacks.append(lr_callbacks)
    tensorboard_callback = None

    if args.tensorboard_dir:
        file_writer = tf.summary.create_file_writer(args.tensorboard_dir)
        file_writer.set_as_default()
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        from eval.pascal import Evaluate
        evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5' if args.compute_val_loss
                else f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}.h5'
            ),
            verbose=1,
            save_weights_only=True,
        )
        callbacks.append(checkpoint)

    return callbacks


def create_generators(args, step1=False):
    common_args = {
        'batch_size': args.step1_batch_size if step1 else args.step2_batch_size,
        'phi': args.phi,
        'detect_quadrangle': args.detect_quadrangle
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        misc_effect = MiscEffect()
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None

    if args.dataset_type == 'pascal':
        from generators.pascal import PascalVocGenerator
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            skip_difficult=True,
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'val',
            skip_difficult=True,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'csv':
        from generators.csv_ import CSVGenerator
        train_generator = CSVGenerator(
            args.annotations_path,
            args.classes_path,
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            **common_args
        )

        if args.val_annotations_path:
            validation_generator = CSVGenerator(
                args.val_annotations_path,
                args.classes_path,
                shuffle_groups=False,
                **common_args
            )
        else:
            validation_generator = None
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def build_optimizer(steps_per_epoch, learning_rate=1e-2, warm_up=False):
    if not warm_up:
        learning_rate = tf.keras.experimental.CosineDecay(learning_rate, steps_per_epoch * 25)

    optimizer = Adam(learning_rate=learning_rate)

    return optimizer


def check_args(parsed_args):
    if parsed_args.gpu and parsed_args.step1_batch_size < len(parsed_args.gpu.split(',')):
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.step1_batch_size,
                                                                                             len(parsed_args.gpu.split(
                                                                                                 ','))))

    return parsed_args


def parse_args(args):
    today = str(date.today())
    parser = argparse.ArgumentParser(description='Simple training script for training a efficientDet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).', type=str,
                               default='datasets/VOC2012')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations_path', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes_path', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations-path',
                            help='Path to CSV file containing annotations for validation (optional).')
    parser.add_argument('--detect-quadrangle', help='If to detect quadrangle.', action='store_true', default=False)

    parser.add_argument('--snapshot', help='Resume training from a snapshot.', default='imagenet')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true',
                        default=True)
    parser.add_argument('--weighted-bifpn', help='Use weighted BiFPN', action='store_true', default=False)

    parser.add_argument('--step1-batch-size', help='Size of the batches.', default=32, type=int)
    parser.add_argument('--step2-batch-size', help='Size of the batches.', default=4, type=int)
    parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).', default='4')
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=100)
    parser.add_argument('--step1-epochs', help='Number of STEP1 epochs to train.', type=int, default=50)
    parser.add_argument('--start_epochs', help='start epochs to train.', type=int, default=0)
    parser.add_argument('--skip-step1', help='start epochs to train.', type=int, default=False)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training',
                        default='checkpoints/{}'.format(today))
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output',
                        default='logs/{}'.format(today))
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                        action='store_false')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--compute-val-loss', default=True, help='Compute validation loss during training', dest='compute_val_loss',
                        action='store_true')

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers', help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.', type=int,
                        default=10)
    print(vars(parser.parse_args(args)))
    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generators
    train_generator, validation_generator = create_generators(args, step1=True)

    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    # optionally choose specific GPU
    if args.gpu:
        # GPU settings
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # use args.gpu
            try:
                print('start with GPU {}'.format(args.gpu))
                tf.config.experimental.set_visible_devices(gpus[int(args.gpu)], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[int(args.gpu)], True)
            except RuntimeError as e:
                print(e)

    model, prediction_model = efficientdet(args.phi,
                                           num_classes=num_classes,
                                           num_anchors=num_anchors,
                                           weighted_bifpn=args.weighted_bifpn,
                                           detect_quadrangle=args.detect_quadrangle
                                           )
    # load pretrained weights
    if args.snapshot:
        if args.snapshot == 'imagenet':
            model_name = 'efficientnet-b{}'.format(args.phi)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
        else:
            print('Loading model, this may take a second...')
            model.load_weights(args.snapshot, by_name=True)

    # freeze backbone layers
    if args.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi]):
            model.layers[i].trainable = False

    if args.gpu and len(args.gpu.split(',')) > 1:
        model = keras.utils.multi_gpu_model(model, gpus=list(map(int, args.gpu.split(','))))

    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        prediction_model,
        validation_generator,
        args,
        step1=True
    )

    if not args.compute_val_loss:
        validation_generator = None
    elif args.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    # start training
    # step1
    if not args.skip_step1:
        # compile model
        model.compile(
            optimizer=build_optimizer(steps_per_epoch=train_generator.size() // args.step1_batch_size, warm_up=True),
            loss={
                'regression': smooth_l1_quad() if args.detect_quadrangle else smooth_l1(),
                'classification': focal()
            }, )

        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.size() // args.step1_batch_size,
            epochs=args.step1_epochs,
            verbose=1,
            callbacks=callbacks,
            workers=args.workers,
            use_multiprocessing=args.multiprocessing,
            max_queue_size=args.max_queue_size,
            validation_data=validation_generator
        )

    model.compile(optimizer=build_optimizer(steps_per_epoch=train_generator.size() // args.step2_batch_size), loss={
        'regression': smooth_l1_quad() if args.detect_quadrangle else smooth_l1(),
        'classification': focal()
    }, )

    # create the callbacks
    callbacks = create_callbacks(
        prediction_model,
        validation_generator,
        args
    )

    # 227, 329, 329, 374, 464, 566, 656
    for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi]):
        model.layers[i].trainable = True

    # create the generators
    train_generator, validation_generator = create_generators(args)

    # step2
    return model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size() // args.step2_batch_size,
        initial_epoch=args.start_epochs,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator
    )


if __name__ == '__main__':
    main()
