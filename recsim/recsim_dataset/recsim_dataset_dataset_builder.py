import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for recsim_dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Have user state explicit in this dataset. Does not pick out final horizon step explicitly. 1 user, 20 docs, 2 topics, 2 slate size, 20 horizon.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        self.num_users = 1
        self.num_docs = 10
        self.num_topics = 2
        self.slate_size = 2
        self.history_length = 10
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'doc_id': tfds.features.Tensor(shape=(self.num_docs,), dtype=tf.int32),
                'doc_topic': tfds.features.Tensor(shape=(self.num_docs,), dtype=tf.int32),
                'doc_quality': tfds.features.Tensor(shape=(self.num_docs,), dtype=tf.float32),
                'doc_features': tfds.features.Tensor(shape=(self.num_docs, self.num_topics), dtype=tf.float32),
                'slate_doc_id': tfds.features.Tensor(shape=(self.history_length, 1, self.slate_size), dtype=tf.int32),
                'slate_doc_topic': tfds.features.Tensor(shape=(self.history_length, 1, self.slate_size), dtype=tf.int32),
                'slate_doc_quality': tfds.features.Tensor(shape=(self.history_length, 1, self.slate_size), dtype=tf.float32),
                'slate_doc_features': tfds.features.Tensor(shape=(self.history_length, 1, self.slate_size, self.num_topics), dtype=tf.float32),
                'choice': tfds.features.Tensor(shape=(self.history_length, 1), dtype=tf.int32),
                'consumed_time': tfds.features.Tensor(shape=(self.history_length, 1), dtype=tf.float32),
                'user_state': tfds.features.Tensor(shape=(self.history_length, 1, self.num_topics), dtype=tf.float32),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # (recsim_dataset): Downloads the data and defines the splits
        path = Path('data_0.1')

        # (recsim_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path=path)
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # (recsim_dataset): Yields (key, example) tuples from the dataset
        id = 0
        for f in path.glob('*.pkl'):
            with (open(f, "rb")) as openfile:
                while True:
                    try:
                        traj = pickle.load(openfile)
                    except EOFError:
                        break

                docs = traj['available docs']
                doc_history = traj['slate docs']
                response_history = traj['user response']
                state_history = traj['user state']
                id += 1

                yield id, {
                    'doc_id': docs.get('doc_id').numpy()[0],
                    'doc_topic': docs.get('doc_topic').numpy()[0],
                    'doc_quality': docs.get('doc_quality').numpy()[0],
                    'doc_features': docs.get('doc_features').numpy()[0],
                    'slate_doc_id': doc_history.get('doc_id').numpy(),
                    'slate_doc_topic': doc_history.get('doc_topic').numpy(),
                    'slate_doc_quality': doc_history.get('doc_quality').numpy(),
                    'slate_doc_features': doc_history.get('doc_features').numpy(),
                    'choice': response_history.get('choice').numpy(),
                    'consumed_time': response_history.get('consumed_time').numpy(),
                    'user_state' : state_history.get('interest.state').numpy(),
                }
