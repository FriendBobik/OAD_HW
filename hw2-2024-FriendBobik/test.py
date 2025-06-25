import unittest
import lsp
import numpy as np

class LspTest(unittest.TestCase):
    def test_lstsq_ne(self):
        a = np.eye(4)
        b = np.ones(a.shape[0])
        x, cost, *_ = lsp.lstsq_ne(a, b)
        np.testing.assert_allclose(x, np.ones(a.shape[1]), rtol=1e-6)
        #np.testing.assert_allclose(cost, np.asarray([0.0]), rtol=1e-6)

    def test_lstsq_svd(self):
        a = np.eye(4)
        b = np.ones(a.shape[0])
        x, cost, *_ = lsp.lstsq_svd(a, b)
        np.testing.assert_allclose(x, np.ones(a.shape[1]), rtol=1e-6)
        #np.testing.assert_allclose(cost, np.asarray([0.0]), rtol=1e-6)

    def test_lstsq_m_ne(self):
        a = np.eye(4)
        b = np.ones(a.shape[0])
        x, cost, *_ = lsp.lstsq(a, b, method="ne")
        np.testing.assert_allclose(x, np.ones(a.shape[1]), rtol=1e-6)
        #np.testing.assert_allclose(cost, np.asarray([0.0]), rtol=1e-6)

    def test_lstsq_m_svd(self):
        a = np.eye(4)
        b = np.ones(a.shape[0])
        x, cost, *_ = lsp.lstsq(a, b, method="svd")
        np.testing.assert_allclose(x, np.ones(a.shape[1]), rtol=1e-6)
        #np.testing.assert_allclose(cost, np.asarray([0.0]), rtol=1e-6)
