import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

from arler.RideHailing.Agent import Taxi


class TaxiCallCentre:
    def __init__(self, name, taxi):
        self.name = name
        self.taxi = taxi
        self.performance = []

    def modelPath(self, path):
        filename = "{}.npz".format(self.name)
        return os.path.join(path, filename)

    def retain(self, path):
        np.savez(
            self.modelPath(path),
            performance=self.performance,
            rewards=self.taxi.completionCost,
            discountedRewards=self.taxi.discountedCompletionCost,
        )

    def recall(self, path):
        modelPath = self.modelPath(path)
        if Path(modelPath).is_file():
            memory = np.load(modelPath)
            self.performance = memory["performance"]
            self.taxi.completionCost = memory["rewards"]
            self.taxi.discountedCompletionCost = memory["discountedRewards"]

    def send(self):
        self.taxi.reset()
        self.taxi.run()
        return self.taxi.score

    def visualise(self, path):
        filename = "{}-LR{}-DF{}-ER{}.png".format(
            self.name,
            self.taxi.learningRate,
            self.taxi.discountFactor,
            self.taxi.explorationRate,
        )
        figpath = os.path.join(path, filename)
        fig = plt.figure()
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.plot(range(len(self.performance)), self.performance)
        plt.savefig(figpath)
        plt.show(fig)

    def run(self, modelsDir, imagesDir, episodes):
        self.recall(modelsDir)
        extra = []
        for _ in range(episodes):
            extra.append(self.send())
        self.performance = np.append(self.performance, extra)
        self.retain(modelsDir)
        self.visualise(imagesDir)
