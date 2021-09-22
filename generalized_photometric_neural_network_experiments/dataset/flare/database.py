from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class FlareDatabase(StandardAndInjectedLightCurveDatabase):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
        # self.training_standard_light_curve_collections = [
        #     TransitExperimentLightCurveCollection(label=TransitLabel.PLANET, splits=list(range(8))),
        #     TransitExperimentLightCurveCollection(label=TransitLabel.NON_PLANET, splits=list(range(8))),
        # ]
        # self.validation_standard_light_curve_collections = [
        #     TransitExperimentLightCurveCollection(splits=[8]),
        # ]
        # self.inference_light_curve_collections = [
        #     TransitExperimentLightCurveCollection(splits=[9]),
        # ]
