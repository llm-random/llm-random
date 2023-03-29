from lizrd.support.test_utils import GeneralTestCase
from lizrd.train.mem_helper import (
    interpret_trainer_scheduler_string,
    create_trainer_scheduler,
)


class TestScheduling(GeneralTestCase):
    def test_basic(self):
        string_accept_reject = [
            ["justonce:start=1:times=2", [1, 2], [0, 3, 4, 5, 8, 10, 1000]],
            [
                "constant:start=1:gap=100",
                [1, 101, 201, 301],
                [0, 2, 3, 4, 5, 8, 10, 1000],
            ],
            [
                "backoff:start=1:exponent=2:initial_gap=1",
                [1, 2, 4, 8, 16],
                [0, 3, 5, 6, 7, 9, 10, 1000],
            ],
            [
                "backoff:start=0:exponent=3",
                [0, 1, 4, 13, 40],
                [2, 3, 5, 6, 7, 8, 9, 10, 1000],
            ],
        ]
        for conf_string, accept, reject in string_accept_reject:
            conf = interpret_trainer_scheduler_string(conf_string)
            scheduler = create_trainer_scheduler(
                scheduler_type=conf["scheduler_type"], **conf["kwargs"]
            )
            for accept_epoch in accept:
                self.assertTrue(scheduler(accept_epoch))
            for reject_epoch in reject:
                self.assertFalse(scheduler(reject_epoch))
