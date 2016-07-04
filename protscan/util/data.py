"""HDF Data Manager."""

import pandas as pd
from eden.converter.graph.node_link_data import node_link_data_to_eden

import logging
logger = logging.getLogger(__name__)

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"


class HDFDataManager():
    """HDF Data Manager."""

    def __init__(self, store_path):
        """Constructor.

        Parameters
        ----------
        store_path : HDFStore
            HDF store containing the folded RNA structures.
        """
        try:
            self.store = pd.io.pytables.HDFStore(store_path)
        except:
            logger.debug("Couldn't open HDF store %s" % store_path)
            exit(1)

    def get_keys(self):
        """Return the keys in the store."""
        return self.store.keys()

    def _ret(self, id_):
        node_link_data = self.store.get(id_)[0]
        return list(node_link_data_to_eden([node_link_data]))[0]

    def retrieve(self, id_list):
        """Retrieve a list of IDs."""
        for id_ in id_list:
            yield self._ret(id_)
        self.store.close()
