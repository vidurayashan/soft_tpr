""" 
Code below adapted from open-source implementation of disentanglement_lib at https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/abstract_reasoning/pgm_data.py
@inproceedings{locatello2019challenging,
  title={Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations},
  author={Locatello, Francesco and Bauer, Stefan and Lucic, Mario and Raetsch, Gunnar and Gelly, Sylvain and Sch{\"o}lkopf, Bernhard and Bachem, Olivier},
  booktitle={International Conference on Machine Learning},
  pages={4114--4124},
  year={2019}
}
"""

import numpy as np
from typing import List
from torch.utils.data import Dataset
from src.data.datasets import DisLibDataset
from src.eval.avr.data import pgm_utils
from src.eval.avr.shared import vis_utils

class Quantizer(DisLibDataset):
  """Quantizes a GroundTruthData to have a maximal number of factors."""

  def __init__(self, wrapped_ground_truth_data: DisLibDataset, max_factors):
    """Constructs a Quantizer.

    Args:
      wrapped_ground_truth_data: GroundTruthData that should be quantized.
      max_factors: integer with the maximal number of factors.
    """
    self.wrapped_ground_truth_data = wrapped_ground_truth_data
    self.true_num_factors = wrapped_ground_truth_data.factor_sizes
    self.fake_num_factors = list(np.minimum(self.true_num_factors, max_factors))

  @property
  def num_factors(self):
    return self.wrapped_ground_truth_data.num_factors

  @property
  def factors_num_values(self):
    return self.fake_num_factors
  
  @property
  def factor_sizes(self): 
     return self.fake_num_factors
  

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    factors = np.zeros(shape=(num, self.num_factors), dtype=np.int64)
    for i in range(self.num_factors):
      factors[:, i] = self._sample_factor(i, num, random_state)
    return factors

  def _sample_factor(self, i, num, random_state):
    return random_state.randint(self.factor_sizes[i], size=num)

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    translated_factors = np.copy(factors)
    for i in range(self.num_factors):
      if self.true_num_factors[i] != self.fake_num_factors[i]:
        ratio = float(self.true_num_factors[i]) / float(
            self.fake_num_factors[i])
        translated_factors[:, i] = np.floor(factors[:, i] * ratio)
    return self.wrapped_ground_truth_data.sample_observations_from_factors(
        translated_factors, random_state)

class PGMDataset(Dataset): 
    def __init__(self, gt_dataset: Quantizer,
                 sampling_strategy: str, relations_dist: List[int]):
        """Creates a PGMDataset.
            Args:
            ground_truth_data: GroundTruthData data set used to generate images.
            sampling_strategy: Either `easy` or `hard`. For `easy`, alternative
                answers are random other solutions that do not satisfy the constraints
                in the given PGM. For `hard`, alternative answers are unique random
                modifications of the correct solution which makes the task  harder.
            relations_dist: List with probabilites where the i-th element contains the
                probability that i relations are enforced.
        """
        self.gt_dataset = gt_dataset 
        self.relations_dist = relations_dist
        self.sampling_strategy = sampling_strategy

    def __len__(self): 
       return 999999 # kind of dodgy

    def sample(self, random_state: np.random.RandomState=None):
        if random_state == None: 
           random_state = np.random.RandomState(np.random.randint(low=0, high=999999)) 
        num_relations = 1 + random_state.choice(
            len(self.relations_dist), p=self.relations_dist
        )
        # construct PGM solution in space of ground-truth factors 
        pgm = pgm_utils.PGM(
            random_state, 
            num_relations, 
            self.gt_dataset.factor_sizes
        )

        # sample instances of images for solutions and alternative answers
        solution = []
        for row in pgm.matrix: 
            solution.append(
                self.gt_dataset.sample_observations_from_factors(
                    row, random_state
                )
            )
        alternatives = self.gt_dataset.sample_observations_from_factors(
            pgm.other_solutions, random_state
        )
        
        # sample position of correct answer
        position = random_state.choice(alternatives.shape[0] + 1)
        # return instance
        return PGMInstance(
            np.array(solution), alternatives, position, pgm.matrix, 
            pgm.other_solutions, self.gt_dataset.factor_sizes
        )
    
    def __getitem__(self, idx):
        instance = self.sample(np.random.RandomState(idx))
        return instance.training_sample()
    

class PGMInstance(object): 
    def __init__(self, solution, alternatives, position, solution_factors=None, 
                 alternatives_factors=None, num_factor_values=None): 
        """Constructs a PGMInstance.
        Args:
        solution: Numpy array of shape (num_rows, num_cols, width, height,
            channels) with the images of the PGM solution.
        alternatives: Numpy array of shape (num_alternatives, width, height,
            channels) with the images of the alternatives.
        position: Integer with position where solution should be inserted.
        solution_factors: Numpy array of shape (num_rows, num_cols, num_factors)
            with the factors of the PGM solution.
        alternatives_factors: Numpy array of shape (num_alternatives, num_factors)
            with the images of the alternatives.
        num_factor_values: List with the number of values for each factor.
        """
        self.solution = solution.transpose(0, 1, 3, 4, 2) 
        self.alternatives = alternatives.transpose(0, 2, 3, 1)
        self.position = position 
        self.solution_factors = solution_factors 
        self.alternatives_factors = alternatives_factors
        self.num_factor_values = num_factor_values

    def get_context(self): 
        """Returns the context: 
        Returns:
        Numpy array of shape (num_rows*num_cols - 1, width, height, channels).
        """
        context = [] 
        for row in self.solution: 
            context += list(row)
        return np.array(context[:-1], dtype=np.float32)

    def get_answers(self): 
        """Returns the answers.
        Returns:
        Numpy array of shape (num_alternatives + 1, width, height, channels).
        """
        result = list(self.alternatives) 
        result.insert(self.position, self.solution[-1, -1])
        return np.array(result, dtype=np.float32)
    
    def get_context_factor_values(self):
        """Returns the context ground truth factos as integer values.

        Returns:
          Numpy array of shape (num_rows*num_cols - 1, len(num_factor_values).
        """
        context = []
        for row in self.solution_factors:
          context += list(row)
        return np.array(context[:-1])

    def get_answers_factor_values(self):
      """Returns the answers ground truth factos as integer values.

      Returns:
        Numpy array of shape (num_alternatives + 1, len(num_factor_values).
      """
      result = list(self.alternatives_factors)
      result.insert(self.position, self.solution_factors[-1, -1])
      return np.array(result)

    def range_embed_factors(self, factors):
        """Embeds the factors linearly in [-0.5, 0.5] based on integer values.

        Args:
          factors: Numpy array of shape (:, len(num_factor_values) with factors.

        Returns:
          Numpy array of shape (:, len(num_factor_values) with floats.
        """
        result = np.array(factors, dtype=np.float32)
        max_vals = np.array(self.num_factor_values, dtype=np.float32) - 1.
        result /= np.expand_dims(max_vals, 0)
        return result - .5

    def onehot_embed_factors(self, factors):
        """Embeds the factors as one-hot vectors.

        Args:
          factors: Numpy array of shape (:, len(num_factor_values) with factors.

        Returns:
          Numpy array of shape (:, sum(num_factor_values) with floats.
        """
        result = []
        for i, num in enumerate(self.num_factor_values):
          result.append(onehot(factors[:, i], num))
        return np.array(np.concatenate(result, axis=-1), dtype=np.float32)
    
    def training_sample(self):
      """Returns a single training example."""
      sample = {}
      sample["context"] = self.get_context()
      sample["answers"] = self.get_answers()
      if self.solution_factors is not None:
        context_factors = self.get_context_factor_values()
        answers_factors = self.get_answers_factor_values()

        sample["context_factor_values"] = self.range_embed_factors(
            context_factors)
        sample["answers_factor_values"] = self.range_embed_factors(
            answers_factors)
        sample["context_factors_onehot"] = self.onehot_embed_factors(
            context_factors)
        sample["answers_factors_onehot"] = self.onehot_embed_factors(
            answers_factors)
      return sample, self.position
    

    def make_image(self, answer=False, padding_px=8, border_px=4):
        """Creates an image of the PGMInstance."""
        # Create the question side that contains the progression matrix.
        question = np.copy(self.solution)
        if question.shape[-1] == 1:
          question = np.repeat(question, 3, -1)
        if not answer:
          question[-1, -1] = question_mark()
    
        # Build up the image on the context side.
        rows = []
        for i in range(question.shape[0]):
          row = []
          for j in range(question.shape[1]):
            # Do the border around the image.
            color = np.array([1., 1., 1.])
            if answer and i == (question.shape[0] - 1) and j == (question.shape[1] -
                                                                 1):
              color = COLORS["green"]
            row.append(
                vis_utils.pad_around(question[i, j], border_px, value=color))
          rows.append(vis_utils.padded_stack(row, padding_px, axis=1))
        question_image = vis_utils.padded_stack(rows, padding_px)
    
        separator = np.zeros((question_image.shape[0], 2, question_image.shape[2]))
    
        # Create the answer side.
        answers = self.get_answers()
        if answers.shape[-1] == 1:
          answers = np.repeat(answers, 3, -1)
        answers_with_border = []
        for i, image in enumerate(answers):
          color = np.array([1., 1., 1.])
          if answer:
            color = COLORS["green"] if i == self.position else COLORS["red"]
          answers_with_border.append(
              vis_utils.pad_around(image, border_px, value=color))
    
        answer_image = vis_utils.padded_grid(answers_with_border,
                                                  question.shape[0], padding_px)
        center_crop = vis_utils.padded_stack(
            [question_image, separator, answer_image], padding_px, axis=1)
        return vis_utils.pad_around(
            vis_utils.add_below(center_crop, padding_px), padding_px)


def onehot(indices, num_atoms):
  """Embeds the indices as one hot vectors."""
  return np.eye(num_atoms)[indices]


from src.shared.constants import *
from src.data.datasets import get_dataset

def get_pgm_dataset(pgm_type: str, dataset: str, data_dir) -> PGMDataset: 
    gt_dataset, number_factors, number_channels, test_ratio_per_factor = \
        get_dataset(dataset, data_dir)

    if dataset == SHAPES3D_DATASET: 
        wrapped_dataset = Quantizer(gt_dataset, [10, 10, 10, 4, 4, 4])
    else: 
       raise NotImplementedError(f"Unknown dataset {dataset}")

    if pgm_type.startswith('easy'): 
       sampling = 'easy'
    elif pgm_type.startswith('hard'): 
       sampling = 'hard'
    else:
       raise ValueError(f'Invalid sampling strategy {sampling}')
    
    if pgm_type.endswith("1"):
        relations_dist = [1., 0., 0.]
    elif pgm_type.endswith("2"):
        relations_dist = [0., 1., 0.]
    elif pgm_type.endswith("3"):
        relations_dist = [0., 0., 1.]
    elif pgm_type.endswith("mixed"):
        relations_dist = [1. / 3., 1. / 3., 1. / 3.]
    else:
        raise ValueError("Invalid number of relations.")

    return PGMDataset(
      wrapped_dataset,
      sampling_strategy=sampling,
      relations_dist=relations_dist)


QUESTION_MARK = [None]

COLORS = {
    "blue": np.array([66., 103., 210.]) / 255.,
    "red": np.array([234., 67., 53.]) / 255.,
    "yellow": np.array([251., 188., 4.]) / 255.,
    "green": np.array([52., 168., 83.]) / 255.,
    "grey": np.array([154., 160., 166.]) / 255.,
}


from PIL import Image
def question_mark():
    """Returns an image of the question mark."""
    # Cache the image so it is not always reloaded.
    if QUESTION_MARK[0] is None:
        with open('/home/bethia/Pictures/qmark.jpg', 'rb') as f:
            QUESTION_MARK[0] = np.array(Image.open(f).convert("RGB").resize((64, 64))) * 1.0 / 255.
    return QUESTION_MARK[0]

