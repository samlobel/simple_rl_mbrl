# Python imports
import numpy as np
from PIL import Image

# Local imports
from simple_rl.mdp.StateClass import State

''' GymStateClass.py: Contains a State class for Gym. '''

class GymState(State):
    ''' Gym State class '''

    def __init__(self, data=[], is_terminal=False, is_time_limit_truncated=False):
        self.data = data
        State.__init__(self, data=data, is_terminal=is_terminal, is_time_limit_truncated=is_time_limit_truncated)

    def features(self):
        return self.data

    def render(self):
        img = Image.fromarray(self.data)
        img.show()
