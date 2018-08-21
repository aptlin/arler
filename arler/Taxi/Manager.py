import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

from .Agent import Taxi


class TaxiCallCentre:
    def __init__(self, name, taxi):
        self.name = name
        self.taxi = taxi
        self.performance = []

    def retain(self):
        np.savez(
            self.name,
            performance=self.performance,
            rewards=self.taxi.completionCost,
            discountedRewards=self.taxi.discountedCompletionCost,
        )

    def recall(self):
        filename = "{}.npz".format(self.name)
        if Path(filename).is_file():
            memory = np.load(filename)
            self.performance = memory["performance"]
            self.taxi.completionCost = memory["rewards"]
            self.taxi.discountedCompletionCost = memory["discountedRewards"]

    def send(self):
        self.taxi.reset()
        self.taxi.run()
        return self.taxi.score

    def visualise(self):
        fig = plt.figure()
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.plot(range(len(self.performance)), self.performance)
        plt.savefig(
            "{}-LR{}-DF{}-ER{}.png".format(
                self.name,
                self.taxi.learningRate,
                self.taxi.discountFactor,
                self.taxi.explorationRate,
            )
        )
        plt.show(fig)

    def run(self, episodes):
        self.recall()
        extra = []
        for _ in range(episodes):
            extra.append(self.send())
        self.performance = np.append(self.performance, extra)
        self.retain()
        self.visualise()
