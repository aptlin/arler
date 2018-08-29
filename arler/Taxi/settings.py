from arler.Utilities.build import buildTrajectory
import json
import os


class Configuration:
    def __init__(self, constantsPath="constants.json"):
        self.constants = self.__init_constants(constantsPath)
        self.customHierarchyBlueprint = self.__init_custom_hierarchy_blueprint()
        self.trajectory = self.__init_trajectory()

    def __init_constants(self, constantsPath="constants.json"):
        settingsDir = os.path.dirname(__file__)
        constantsPath = os.path.join(settingsDir, constantsPath)
        with open(constantsPath) as data:
            try:
                constants = json.load(data)
                return constants
            except ValueError:
                raise ValueError("Could not read constants, aborting")

    def __init_custom_hierarchy_blueprint(self):
        goto = self.constants["primitives"]["navigation"]
        pickup = self.constants["primitives"]["service"]["PICKUP"]
        dropoff = self.constants["primitives"]["service"]["DROPOFF"]

        blueprint = {
            "ROOT": {
                "GET": {
                    "PICKUP": pickup,
                    "GOTO_R": goto,
                    "GOTO_G": goto,
                    "GOTO_B": goto,
                    "GOTO_Y": goto,
                },
                "PUT": {
                    "DROPOFF": dropoff,
                    "GOTO_R": goto,
                    "GOTO_G": goto,
                    "GOTO_B": goto,
                    "GOTO_Y": goto,
                },
            }
        }
        return blueprint

    def __init_trajectory(self):
        blueprint = self.constants["trajectory"]
        return buildTrajectory(blueprint)
