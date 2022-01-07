
#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--use_latest_checkpoint', action='store_true', help='if set, inference will use last saved epoch, otherwise it will use best validation performing model')

        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser