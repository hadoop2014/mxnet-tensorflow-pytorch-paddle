import mxnet.gluon.data as gdata
import os

class HOTDOG(gdata.vision.ImageFolderDataset):
    """HOTDOG handwritten digits dataset from https://apache-mxnet.s3-accelerate.amazonaws.com/

    Each sample is an image (in 3D NDArray) with shape (?, ?, 3).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/hotdog
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    from mxnet import base
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'hotdog'),
                 train=True, transform=None):
        self._hotdog_data = ('hotdog.zip',
                             'fba480ffa8aa7e0febbb511d181409f899b9baa5')
        self._download_data(root)
        if train:
            self._root = os.path.join(root, 'hotdog/train')
        else:
            self._root = os.path.join(root, 'hotdog/test')
        self.flag = 1  # flag=1指转换为3通道图片
        super(HOTDOG, self).__init__(self._root, flag=self.flag, transform=transform)

    def _download_data(self, root):
        from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
        import zipfile
        data = self._hotdog_data
        namespace = 'gluon/dataset'
        data_file = download(_get_repo_file_url(namespace, data[0]),
                             path=root,
                             sha1_hash=data[1])
        with zipfile.ZipFile(data_file, 'r') as z:
            z.extractall(root)
