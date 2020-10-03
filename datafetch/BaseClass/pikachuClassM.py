import mxnet.gluon.data as gdata
from mxnet import image
import os

class PIKACHU():
    """PIKACHU handwritten digits dataset from https://apache-mxnet.s3-accelerate.amazonaws.com/

    Each sample is an image (in 3D NDArray) with shape (?, ?, 3).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/PIKACHU
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    from mxnet import base
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'pikachu'),
                 train=True, transform=None):
        self._train=train
        self._root = root
        self._train_rec = ('train.rec','e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8')
        self._train_idx = ('train.idx', 'dcf7318b2602c06428b9988470c731621716c393')
        self._test_rec = ('val.rec', 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520')
        self._namespace='pikachu'
        self._download_data(root)
        #self.flag = 1  # flag=1指转换为3通道图片
        #super(PIKACHU, self).__init__(self._root, flag=self.flag, transform=transform)

    def _download_data(self, root):
        from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
        namespace = 'gluon/dataset/' + self._namespace
        if self._train == True:
            data = self._train_rec
            data_file = download(_get_repo_file_url(namespace, data[0]),
                                 path=root,
                                 sha1_hash=data[1])
            self.path_rec = data_file
            data = self._train_idx
            data_file = download(_get_repo_file_url(namespace, data[0]),
                                 path=root,
                                 sha1_hash=data[1])
            self.path_idx = data_file
        else:
            data = self._test_rec
            data_file = download(_get_repo_file_url(namespace, data[0]),
                                 path=root,
                                 sha1_hash=data[1])
            self.path_rec = data_file
            self.path_idx = None
