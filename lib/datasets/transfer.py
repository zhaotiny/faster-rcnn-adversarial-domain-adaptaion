import datasets.imdb
from enum import Enum

class Domain(Enum):
    SOURCE="source"
    TARGET="target"

class transfer(datasets.imdb):
    def __init__(self, source_imdb, target_imdb):
        self.name = source_imdb.name + "_to_" + target_imdb.name
        self._source_imdb = source_imdb
        self._target_imdb = target_imdb

    def get_imdb(self, domain):
        if domain is Domain.SOURCE:
            return self._source_imdb
        elif domain is Domain.TARGET:
            return self._target_imdb
        else:
            raise TypeError("domain must be either Domain.SOURCE or Domain.TARGET")

    def image_path_at(self, i, domain=Domain.SOURCE):
        """
        Return the absolute path to image i in the domain image sequence.
        """
        imdb = self.get_imdb(domain)
        return imdb.image_path_at(i)

    def image_path_from_index(self, index, domain=Domain.SOURCE):
        """
        Construct an image path from the image's "index" identifier.
        """
        imdb = self.get_imdb(domain)
        return imdb.image_path_from_index(index)

    def gt_roidb(self, domain=Domain.SOURCE):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        imdb = self.get_imdb(domain)
        return imdb.gt_roidb()
