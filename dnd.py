import utils.kdtree as kdtree

from torch import Tensor
from torch.autograd import Variable
from utils.lru_cache import LRUCache

class DND:
  def __init__(self, kernel, num_neighbors, max_memory, embedding_size):
    self.dictionary = LRUCache(max_memory)
    self.kd_tree = kdtree.create(dimensions=embedding_size)

    self.num_neighbors = num_neighbors
    self.kernel = kernel
    self.max_memory = max_memory

  def is_present(self, key):
    return self.dictionary.get(tuple(key.data.cpu().numpy()[0])) is not None

  def get_value(self, key):
    return self.dictionary.get(tuple(key.data.cpu().numpy()[0]))

  def lookup(self, lookup_key):
    keys = [key[0].data for key in self.kd_tree.search_knn(lookup_key, self.num_neighbors)]
    output, kernel_sum = 0, 0
    for key in keys:
      if not (key == lookup_key).data.all():
        output += self.kernel(key, lookup_key) * self.dictionary.get(tuple(key.data.cpu().numpy()[0]))
        kernel_sum += self.kernel(key, lookup_key)
    output = output / kernel_sum
    return output

  def upsert(self, key, value):
    if not self.is_present(key):
      self.kd_tree.add(key)

      if self.dictionary.size() == self.max_memory:
        # Expel least recently used key from self.dictionary and self.kd_tree if memory used is at capacity
        deleted_key = self.dictionary.delete_least_recently_used()
        self.kd_tree.remove(Variable(Tensor(list(deleted_key))).unsqueeze(0))

    self.dictionary.set(tuple(key.data.cpu().numpy()[0]), value)
