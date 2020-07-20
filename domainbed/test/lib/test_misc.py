# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from domainbed.lib import misc

class TestMisc(unittest.TestCase):

    def test_make_weights_for_balanced_classes(self):
        dataset = [('A', 0), ('B', 1), ('C', 0), ('D', 2), ('E', 3), ('F', 0)]
        result = misc.make_weights_for_balanced_classes(dataset)
        self.assertEqual(result.sum(), 1)
        self.assertEqual(result[0], result[2])
        self.assertEqual(result[1], result[3])
        self.assertEqual(3 * result[0], result[1])
