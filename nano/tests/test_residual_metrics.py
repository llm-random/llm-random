import argparse
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from hydra import initialize, compose


from model import (
    RecorderLogger,
    run,
)

logger = logging.getLogger(__name__)

target_residual_metrics = {
    "steps/block_0/residual_attention/update_norms/mean": [
        [3.610635280609131, 0],
        [3.384478807449341, 1],
        [3.5041885375976562, 2],
        [3.385528564453125, 3],
        [3.4919495582580566, 4],
        [3.39223313331604, 5],
        [3.24385404586792, 6],
        [3.5691239833831787, 7],
        [3.426798105239868, 8],
        [3.292729377746582, 9],
    ],
    "steps/block_0/residual_attention/update_norms/std": [
        [0.8251062035560608, 0],
        [0.9845030307769775, 1],
        [0.7598809003829956, 2],
        [0.8928273320198059, 3],
        [0.7844108939170837, 4],
        [1.0856202840805054, 5],
        [0.7877475619316101, 6],
        [0.600958526134491, 7],
        [0.8503165245056152, 8],
        [0.7899901270866394, 9],
    ],
    "steps/block_0/residual_attention/residual_norms/mean": [
        [7.609645843505859, 0],
        [7.474404811859131, 1],
        [7.243101596832275, 2],
        [7.505288124084473, 3],
        [7.076922416687012, 4],
        [7.4320831298828125, 5],
        [7.886249542236328, 6],
        [7.703823089599609, 7],
        [7.452531337738037, 8],
        [7.318223476409912, 9],
    ],
    "steps/block_0/residual_attention/residual_norms/std": [
        [0.7482208609580994, 0],
        [0.6164348721504211, 1],
        [0.7875577807426453, 2],
        [0.6930263042449951, 3],
        [0.7000119686126709, 4],
        [0.7768588066101074, 5],
        [0.5001723766326904, 6],
        [0.7697810530662537, 7],
        [0.9729623198509216, 8],
        [0.9434953331947327, 9],
    ],
    "steps/block_0/residual_attention/update_to_residual_ratio/mean": [
        [0.47911906242370605, 0],
        [0.4512171447277069, 1],
        [0.4833788573741913, 2],
        [0.45124709606170654, 3],
        [0.5013516545295715, 4],
        [0.45653218030929565, 5],
        [0.41196078062057495, 6],
        [0.4656982123851776, 7],
        [0.4628787338733673, 8],
        [0.4503737688064575, 9],
    ],
    "steps/block_0/residual_attention/update_to_residual_ratio/std": [
        [0.12213237583637238, 0],
        [0.11765696108341217, 1],
        [0.08517482876777649, 2],
        [0.11014211922883987, 3],
        [0.13732489943504333, 4],
        [0.13508635759353638, 5],
        [0.09892719238996506, 6],
        [0.07874521613121033, 7],
        [0.11100264638662338, 8],
        [0.08657049387693405, 9],
    ],
    "steps/block_0/residual_feedforward/update_norms/mean": [
        [2.98101806640625, 0],
        [2.8771538734436035, 1],
        [2.9332618713378906, 2],
        [3.18682861328125, 3],
        [2.8913216590881348, 4],
        [3.19130802154541, 5],
        [3.0597991943359375, 6],
        [2.916193723678589, 7],
        [2.9807162284851074, 8],
        [3.2072432041168213, 9],
    ],
    "steps/block_0/residual_feedforward/update_norms/std": [
        [0.4167124032974243, 0],
        [0.2925933301448822, 1],
        [0.4983386993408203, 2],
        [0.555364727973938, 3],
        [0.5185657739639282, 4],
        [0.5941171050071716, 5],
        [0.3420737683773041, 6],
        [0.4145314693450928, 7],
        [0.5530945062637329, 8],
        [0.5853129625320435, 9],
    ],
    "steps/block_0/residual_feedforward/residual_norms/mean": [
        [8.59775447845459, 0],
        [8.703665733337402, 1],
        [8.361714363098145, 2],
        [8.473139762878418, 3],
        [8.313697814941406, 4],
        [8.413660049438477, 5],
        [8.703521728515625, 6],
        [8.828185081481934, 7],
        [8.583868026733398, 8],
        [8.334815979003906, 9],
    ],
    "steps/block_0/residual_feedforward/residual_norms/std": [
        [0.9284675121307373, 0],
        [0.7848997712135315, 1],
        [1.2436518669128418, 2],
        [1.2374969720840454, 3],
        [0.7517852187156677, 4],
        [1.2701083421707153, 5],
        [0.7098544239997864, 6],
        [0.9436455965042114, 7],
        [1.3935602903366089, 8],
        [1.2613369226455688, 9],
    ],
    "steps/block_0/residual_feedforward/update_to_residual_ratio/mean": [
        [0.35066133737564087, 0],
        [0.3341410458087921, 1],
        [0.35764822363853455, 2],
        [0.38420361280441284, 3],
        [0.351914644241333, 4],
        [0.38874226808547974, 5],
        [0.35411280393600464, 6],
        [0.3353497087955475, 7],
        [0.35649219155311584, 8],
        [0.3936288356781006, 9],
    ],
    "steps/block_0/residual_feedforward/update_to_residual_ratio/std": [
        [0.06393907964229584, 0],
        [0.050692930817604065, 1],
        [0.07081012427806854, 2],
        [0.08327585458755493, 3],
        [0.07894150912761688, 4],
        [0.0959344208240509, 5],
        [0.051131296902894974, 6],
        [0.06826400756835938, 7],
        [0.08717071264982224, 8],
        [0.09629654139280319, 9],
    ],
    "steps/block_1/residual_attention/update_norms/mean": [
        [3.7302026748657227, 0],
        [3.301858425140381, 1],
        [3.423598289489746, 2],
        [3.5384018421173096, 3],
        [3.0697121620178223, 4],
        [3.135007619857788, 5],
        [3.4420042037963867, 6],
        [3.5292558670043945, 7],
        [3.3380002975463867, 8],
        [3.194458484649658, 9],
    ],
    "steps/block_1/residual_attention/update_norms/std": [
        [0.9303711652755737, 0],
        [0.9879388809204102, 1],
        [0.7966120839118958, 2],
        [1.1059470176696777, 3],
        [0.8264631032943726, 4],
        [0.8269519805908203, 5],
        [0.9627777338027954, 6],
        [1.072964072227478, 7],
        [0.7996673583984375, 8],
        [0.8447416424751282, 9],
    ],
    "steps/block_1/residual_attention/residual_norms/mean": [
        [9.084867477416992, 0],
        [9.210664749145508, 1],
        [8.741632461547852, 2],
        [8.947407722473145, 3],
        [8.654755592346191, 4],
        [8.979574203491211, 5],
        [9.250950813293457, 6],
        [9.207058906555176, 7],
        [9.111549377441406, 8],
        [8.826494216918945, 9],
    ],
    "steps/block_1/residual_attention/residual_norms/std": [
        [0.7168177366256714, 0],
        [0.7473440170288086, 1],
        [1.1558692455291748, 2],
        [1.1502115726470947, 3],
        [0.7957513928413391, 4],
        [0.7966441512107849, 5],
        [0.6882650852203369, 6],
        [1.0719070434570312, 7],
        [1.1208105087280273, 8],
        [0.9034557938575745, 9],
    ],
    "steps/block_1/residual_attention/update_to_residual_ratio/mean": [
        [0.41023051738739014, 0],
        [0.35759150981903076, 1],
        [0.39662420749664307, 2],
        [0.4003884196281433, 3],
        [0.35498642921447754, 4],
        [0.34690678119659424, 5],
        [0.3730970621109009, 6],
        [0.38265499472618103, 7],
        [0.369202584028244, 8],
        [0.36224228143692017, 9],
    ],
    "steps/block_1/residual_attention/update_to_residual_ratio/std": [
        [0.09445220977067947, 0],
        [0.09847543388605118, 1],
        [0.0994933545589447, 2],
        [0.12771499156951904, 3],
        [0.09012852609157562, 4],
        [0.07317285984754562, 5],
        [0.10053180903196335, 6],
        [0.10390160977840424, 7],
        [0.09265584498643875, 8],
        [0.09097002446651459, 9],
    ],
    "steps/block_1/residual_feedforward/update_norms/mean": [
        [3.244837760925293, 0],
        [3.2800240516662598, 1],
        [3.300290107727051, 2],
        [3.1668317317962646, 3],
        [3.3478004932403564, 4],
        [3.0806524753570557, 5],
        [3.2874441146850586, 6],
        [3.2842185497283936, 7],
        [3.3464431762695312, 8],
        [3.3352773189544678, 9],
    ],
    "steps/block_1/residual_feedforward/update_norms/std": [
        [0.24985843896865845, 0],
        [0.27877917885780334, 1],
        [0.3592231869697571, 2],
        [0.4528467059135437, 3],
        [0.33675625920295715, 4],
        [0.39283275604248047, 5],
        [0.4717734754085541, 6],
        [0.364301860332489, 7],
        [0.3409566879272461, 8],
        [0.28332406282424927, 9],
    ],
    "steps/block_1/residual_feedforward/residual_norms/mean": [
        [9.632333755493164, 0],
        [9.654722213745117, 1],
        [8.998979568481445, 2],
        [9.454333305358887, 3],
        [8.855104446411133, 4],
        [8.975494384765625, 5],
        [9.719091415405273, 6],
        [9.49972152709961, 7],
        [9.259634971618652, 8],
        [9.265551567077637, 9],
    ],
    "steps/block_1/residual_feedforward/residual_norms/std": [
        [0.7030551433563232, 0],
        [0.6729722619056702, 1],
        [0.9134228825569153, 2],
        [0.8592355847358704, 3],
        [0.9643691778182983, 4],
        [0.5599817633628845, 5],
        [0.717577338218689, 6],
        [1.4047335386276245, 7],
        [0.949476957321167, 8],
        [0.9195491671562195, 9],
    ],
    "steps/block_1/residual_feedforward/update_to_residual_ratio/mean": [
        [0.3385401666164398, 0],
        [0.3406963348388672, 1],
        [0.3690599203109741, 2],
        [0.33774879574775696, 3],
        [0.3810487985610962, 4],
        [0.3454110324382782, 5],
        [0.33959609270095825, 6],
        [0.3553270399570465, 7],
        [0.3651203513145447, 8],
        [0.36429545283317566, 9],
    ],
    "steps/block_1/residual_feedforward/update_to_residual_ratio/std": [
        [0.035803522914648056, 0],
        [0.03069116920232773, 1],
        [0.045254625380039215, 2],
        [0.05732657387852669, 3],
        [0.04783600568771362, 4],
        [0.05717429518699646, 5],
        [0.0502048023045063, 6],
        [0.07991770654916763, 7],
        [0.053152766078710556, 8],
        [0.05668321251869202, 9],
    ],
}


class TestResidual(unittest.TestCase):
    mock_hydra_config = SimpleNamespace(
        runtime=SimpleNamespace(output_dir="mock_output_dir_path"),
        output_subdir="mock_subdir_path",
        overrides=SimpleNamespace(task=["test_residual"]),
    )

    target_losses = [
        (12.176456451416016, 0),
        (12.517879486083984, 1),
        (11.428616523742676, 2),
        (11.448310852050781, 3),
        (11.676277160644531, 4),
        (11.996905326843262, 5),
        (11.710165977478027, 6),
        (11.50185775756836, 7),
        (12.7047758102417, 8),
        (11.960006713867188, 9),
    ]
    grad_norms = [
        (5.038923263549805, 0),
        (5.757411479949951, 1),
        (5.018430709838867, 2),
        (5.112786769866943, 3),
        (5.331805229187012, 4),
        (4.8101959228515625, 5),
        (5.520748138427734, 6),
        (5.246978282928467, 7),
        (5.360789775848389, 8),
        (5.46716833114624, 9),
    ]

    def compare_almost_equal_lists(self, list_a, list_b, places: int):
        for a, b in zip(list_a, list_b):
            a_value, a_step = a
            b_value, b_step = b
            self.assertEqual(a_step, b_step)
            self.assertAlmostEqual(a_value, b_value, places=places)

    def run_steps(self, cfg, metric_logger):
        run(cfg)

        self.compare_almost_equal_lists(
            self.target_losses, metric_logger.data["steps/train/loss"], places=5
        )
        self.compare_almost_equal_lists(
            self.grad_norms, metric_logger.data["steps/train/grad_norm"], places=5
        )
        for metric in target_residual_metrics:
            self.compare_almost_equal_lists(
                target_residual_metrics[metric], metric_logger.data[metric], places=5
            )

    @patch("model.get_metric_logger", return_value=RecorderLogger())
    @patch("model.get_composition_file_path", return_value="mock_file.yaml")
    def test_residual_grad_acc_1(self, _mock, get_metric_logger):
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(
                config_name="test_residual",
                overrides=["training.gradient_accumulation_steps=1"],
            )
        metric_logger = get_metric_logger()
        self.run_steps(cfg, metric_logger)

    @patch("model.get_metric_logger", return_value=RecorderLogger())
    @patch("model.get_composition_file_path", return_value="mock_file.yaml")
    def test_residual_grad_acc_2(self, _mock, get_metric_logger):
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(
                config_name="test_residual",
                overrides=["training.gradient_accumulation_steps=2"],
            )
        metric_logger = get_metric_logger()
        self.run_steps(cfg, metric_logger)


def run_chosen_test(args):
    suite = unittest.TestSuite()
    if args.test == "grad_acc_1":
        suite.addTest(TestResidual("test_residual_grad_acc_1"))
    elif args.test == "grad_acc_2":
        suite.addTest(TestResidual("test_residual_grad_acc_2"))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        default=None,
        choices=["grad_acc_1", "grad_acc_2"],
        help="Specify which test to run; if omitted, runs all unittest tests.",
    )
    args = parser.parse_args()

    if args.test is None:
        # No argument → run all tests in this file
        unittest.main(argv=["ignored", "-v"], exit=False)
    else:
        run_chosen_test(args)
