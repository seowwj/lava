# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import traceback
import unittest
import psutil
import os
import time
import numpy as np
from functools import partial
from enum import Enum

from message_infrastructure import Actor
from message_infrastructure import ActorStatus
from message_infrastructure.multiprocessing import MultiProcessing

import time


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize

class Builder():
    def build(self, i):
        time.sleep(0.0001)

def target_fn(*args, **kwargs):
    """
    Function to build and attach a system process to

    :param args: List Parameters to be passed onto the process
    :param kwargs: Dict Parameters to be passed onto the process
    :return: None
    """
    try:
        actor = args[0]
        builder = kwargs.pop("builder")
        idx = kwargs.pop("idx")
        builder.build(idx)
        return 0
    except Exception as e:
        print("Encountered Fatal Exception: " + str(e))
        print("Traceback: ")
        print(traceback.format_exc())
        raise e

class TestMultiprocessing(unittest.TestCase):
    
    def test_multiprocessing_run(self):
        """
        Spawns an actor.
        Checks that an actor is spawned successfully.
        """
        mp = MultiProcessing()
        mp.start()
        builder = Builder()
        for i in range(5):
            bound_target_fn = partial(target_fn, idx=i)
            mp.build_actor(bound_target_fn, builder)

        time.sleep(5)

        for actor in mp.actors:
            self.assertEqual(actor.get_status(), 0)

        mp.stop(True)

    def test_multiprocessing_pause(self):
        """
        Spawns an actor and pause the actor.
        Checks that actor is paused successfully and able to resume running.
        """
        mp = MultiProcessing()
        mp.start()
        builder = Builder()
        for i in range(5):
            bound_target_fn = partial(target_fn, idx=i)
            mp.build_actor(bound_target_fn, builder)

        time.sleep(5)

        for actor in mp.actors:
            actor.status_paused()
            self.assertEqual(actor.get_status(), 1)

        for actor in mp.actors:
            actor.status_running()
            self.assertEqual(actor.get_status(), 0)

        mp.stop(True)

    def test_multiprocessing_shutdown(self):
        """
        Spawns an actor and sends a stop signal.
        Checks that actor is stopped successfully.
        """
        mp = MultiProcessing()
        mp.start()
        builder = Builder()
        for i in range(5):
            bound_target_fn = partial(target_fn, idx=i)
            mp.build_actor(bound_target_fn, builder)

        time.sleep(5)

        for actor in mp.actors:
            actor.status_stopped()
            self.assertEqual(actor.get_status(), 2)

        mp.stop(True)
   
    def test_get_actor_list(self):
        """
        Gets list of actors
        Checks that all actors are of Actor type
        """
        mp = MultiProcessing()
        mp.start()
        builder = Builder()
        for i in range(5):
            bound_target_fn = partial(target_fn, idx=i)
            mp.build_actor(bound_target_fn, builder)

        time.sleep(5)

        for actor in mp.actors:
            self.assertIsInstance(actor, Actor)

        mp.stop(True)

# Run unit tests
if __name__ == '__main__':
    unittest.main()
