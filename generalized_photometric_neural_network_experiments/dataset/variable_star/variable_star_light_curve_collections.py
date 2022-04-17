from typing import List, Union
from peewee import Select

from generalized_photometric_neural_network_experiments.dataset.variable_star.variable_star_metadata_manager import \
    VariableStarMetadata, VariableTypeName
from ramjet.data_interface.tess_ffi_light_curve_metadata_manager import TessFfiLightCurveMetadata
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection


class RrLyraeFfiLightCurveCollection(TessFfiLightCurveCollection):
    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 1

    def get_sql_query(self) -> Select:
        query = super().get_sql_query()
        rr_lyrae_tic_id_query = VariableStarMetadata.select(VariableStarMetadata.tic_id).where(
            VariableStarMetadata.variable_type_name == VariableTypeName.RR_LYRAE.value)
        query = query.where(TessFfiLightCurveMetadata.tic_id.in_(rr_lyrae_tic_id_query))
        return query


class NonRrLyraeVariableFfiLightCurveCollection(TessFfiLightCurveCollection):
    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 0

    def get_sql_query(self) -> Select:
        query = super().get_sql_query()
        non_rr_lyrae_tic_id_query = VariableStarMetadata.select(VariableStarMetadata.tic_id).where(
            VariableStarMetadata.variable_type_name != VariableTypeName.RR_LYRAE.value)
        query = query.where(TessFfiLightCurveMetadata.tic_id.in_(non_rr_lyrae_tic_id_query))
        return query

class NonRrLyraeFfiLightCurveCollection(TessFfiLightCurveCollection):
    def __init__(self, dataset_splits: Union[List[int], None] = None,
                 magnitude_range: (Union[float, None], Union[float, None]) = (None, None)):
        super().__init__(dataset_splits=dataset_splits, magnitude_range=magnitude_range)
        self.label = 0

    def get_sql_query(self) -> Select:
        query = super().get_sql_query()
        rr_lyrae_tic_id_query = VariableStarMetadata.select(VariableStarMetadata.tic_id).where(
            VariableStarMetadata.variable_type_name == VariableTypeName.RR_LYRAE.value)
        query = query.where(TessFfiLightCurveMetadata.tic_id.not_in(rr_lyrae_tic_id_query))
        return query
