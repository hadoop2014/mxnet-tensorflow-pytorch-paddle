import unittest
from datafetch import getConfig
#getConfig = __import__('datafetch.getConfig',fromlist=['getConfig'])

class MyTestCase(unittest.TestCase):

    def test_getPikachu(self):
        from datafetch import getPikachu
        from datafetch import commFunction
        import matplotlib.pyplot as plt
        gConfig = getConfig.get_config()
        getDataClass = getPikachu.create_model(gConfig)
        train_iter = getDataClass.train_iter
        test_iter = getDataClass.test_iter
        batch = train_iter.next()
        edge_size = 256
        print(batch.data[0].shape, batch.label[0].shape)
        imgs = (batch.data[0][0:10].transpose((0, 2, 3, 1))) / 255
        axes = commFunction.show_images(imgs, 2, 5).flatten()
        for ax, label in zip(axes, batch.label[0][0:10]):
            commFunction.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
        plt.show()
        batch = test_iter.next()
        print(batch.data[0].shape, batch.label[0].shape)

if __name__ == '__main__':
    unittest.main()
